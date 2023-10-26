import copy
import numpy as np
import os
from peft import TaskType
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from torchvision import transforms
from typing import List

import model.LAMM.conversations as conversations
from model.LAMM.openlamm import StoppingCriteriaList, LAMMStoppingCriteria, \
    build_one_instance, LAMMPEFTModel, VISION_TAGS
from .moe.layer import MoeLoraLayer, Top2Gating
from .moe import MoeLoraConfig
from .resampler3d import Resampler3D


def make_prompt_start(
    use_system,
    vision_type: List[str], 
    task_type: List[str],
    template=conversations.default_conversation
) -> List[str]:
    PROMPT_START = [
        f'{template.sep} {template.roles[0]}: {VISION_TAGS["sov"][vision_type_i]}'
        for vision_type_i in vision_type
    ]

    if use_system:
        if task_type == "normal":
            return [
                f"{template.system}\n\n" + prompt_start_i 
                for prompt_start_i in PROMPT_START
            ]
        else:
            if template.sys_temp is None:
                return [
                    f"{conversations.conversation_dict[task]}\n\n" + PROMPT_START[i]
                    for i, task in enumerate(task_type)
                ]
            else:
                return [
                    template.sys_temp.format(system_message=conversations.conversation_dict[task]) + PROMPT_START[i]
                    for i, task in enumerate(task_type)
                ]

    else:
        return PROMPT_START


def process_batch_instance(
    tokenizer, 
    batch_of_conversations, 
    max_tgt_len, 
    vision_type,
    template=conversations.default_conversation,
):
    batch_input_ids, batch_target_ids = [], []
    for i, conversation in enumerate(batch_of_conversations):
        _, one_input_ids, one_target_ids = build_one_instance(
            tokenizer, 
            conversation, 
            vision_type=vision_type[i], 
            template=template,
        )
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(
        batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    target_ids = rnn.pad_sequence(
        batch_target_ids, batch_first=True, padding_value=-100
    )
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


DET_ANSWER_TEMPLATE = [
    "The {P} position of the image contains an object that can be classified as {C}.",
    "The object present at the {P} coordinate in the image is classified as {C}.",
    "There is an object at the {P} location of the image that can be identified as belonging to the category of {C}.",
    "At the {P} position of the image, there is an object categorized as {C}.",
    "At the {P} of the image, there is an item that falls under the category of {C}.",
    "At the coordinates of {P} position of the image, there exists an object categorized as {C}.",
    "The {P} position of the image features an object that falls under the category of {C}.",
    'There is an object at the {P} position of the image, and its category is {C}.',
    'Upon close inspection of the image, it can be observed that there is an object positioned at {P} that belongs to the {C} category.',
    'At the exact coordinates of {P} in the image, there is an object that can be identified as belonging to the {C} category, and this object stands out from the rest of the objects in the image due to its unique color and pattern.',
    'Scanning through the image, it becomes evident that there is an object at {P} that falls under the {C} category.',
    'By carefully examining the image, one can spot an object at {P} that belongs to the {C} category.',
    'Positioned at {P} within the image is an object that can be classified as belonging to the {C} category, and this object is also the only one in the image that has a specific type of texture and a distinctive shape that sets it apart from the other objects.',
    'Upon careful examination of the image, it can be observed that there is an object positioned precisely at {P} that falls under the {C} category, and this object is also the only one in the image that has a specific type of pattern or design that makes it stand out from the rest of the objects.'
]


class Octavius(LAMMPEFTModel):

    def __init__(self, **args):
        args['vision_type'] = args['octavius_modality']
        assert len(args['vision_type']) > 0
        super().__init__(**args)

        print(f'Octavius Modalities: {self.vision_type}')

        # 2d modality
        if 'image' in self.vision_type:
            self.llama_proj = nn.Linear(
                self.vision_hidden_size, self.llama_model.config.hidden_size
            )
            print("Octavius 2D projection layer initialized.")

        # 3d modality
        if 'pcl' in self.vision_type:
            self.resampler_3d = Resampler3D(
                args['num_query_rsp_3d'],
                args['num_heads_rsp_3d'],
                args['num_layers_rsp_3d'],
                args['hidden_size_rsp_3d']
            )
            self.pos_3d_proj = nn.Sequential(
                nn.Linear(6, 256),
                nn.ReLU(),
                nn.Linear(256, args['hidden_size_rsp_3d'])
            )
            self.llama_proj_3d = nn.Sequential(
                nn.Linear(args['hidden_size_rsp_3d'], 5120),
                nn.LayerNorm(5120),
            )
            print('Octavius 3D Encoder & projection layer initialized.')

        # LoRA-MoE
        self.gate_mode = self.args['moe_gate_mode']
        assert self.gate_mode in ['top2_gate']
        self.num_experts = self.args['moe_lora_num_experts']
        
        self.gating_network = Top2Gating(
            self.llama_model.config.hidden_size, self.num_experts
        )

        self.device = torch.cuda.current_device()

    def build_projection_layer(self):
        pass

    def build_peft_config(self):
        print(f'Build PEFT model with LoRA-MoE.')
        peft_config = MoeLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args["lora_r"],
            num_experts=self.args["moe_lora_num_experts"],
            gate_mode=self.args["moe_gate_mode"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=self.args["lora_target_modules"],
        )
        return peft_config

    def get_special_tokens(self):
        tokens = []

        for modality in self.vision_type:
            sov = VISION_TAGS['sov'][modality]
            eov = VISION_TAGS['eov'][modality]
            tokens.extend([sov, eov])
            print(f"Add VISION TAG (\"{sov}\" and \"{eov}\") for modality {modality}.")
        return tokens

    def encode_image(self, images):
        with torch.no_grad():
            vision_embeds_2d = self.visual_encoder.forward_patch_features(images)[
                :, : self.num_vision_token
            ]  # bsz x self.num_vision_token x 1024
        vision_embeds_2d = vision_embeds_2d.reshape(-1, self.vision_hidden_size).to(
            self.llama_model.dtype
        )  # bsz*num vision token x 1024
        vision_embeds_2d = self.llama_proj(vision_embeds_2d).reshape(
            -1, self.num_vision_token, self.llama_model.config.hidden_size
        )  # bsz x num_vision_token x llama_size
        return vision_embeds_2d

    def encode_pcl(self, inputs):
        task_type = inputs["task_type"]
        mask = inputs["mask"].to(self.device)

        ref_object_feat_3d = inputs['vision_embeds_3d_ref'].to(self.device)
        ref_object_pos_3d = inputs['vision_pos_3d_ref'].to(self.device)

        vision_embeds_3d_scene_prop = inputs['vision_embeds_3d_scene_prop'].to(self.device)
        vision_pos_3d_scene_prop = inputs['vision_pos_3d_scene_prop'].to(self.device)
    
        # use instance-level feature for classification task and description task
        for i, task_type_i in enumerate(task_type):
            if task_type_i == 'classification3d' or task_type_i == 'description3d':
                vision_embeds_3d_scene_prop[i] = torch.zeros((vision_embeds_3d_scene_prop.shape[1], ref_object_feat_3d[i].shape[0]), dtype=torch.float16).to(self.device)
                vision_embeds_3d_scene_prop[i, 0] = ref_object_feat_3d[i]
                vision_pos_3d_scene_prop[i] = torch.zeros((vision_embeds_3d_scene_prop.shape[1], ref_object_pos_3d[i].shape[0]), dtype=torch.float16).to(self.device)
                vision_pos_3d_scene_prop[i, 0] = ref_object_pos_3d[i]
                mask[i, :] = 0
                mask[i, 0] = 1

        vision_embeds_3d = vision_embeds_3d_scene_prop  # Bsz x N token x C
        vision_embeds_3d_pos = vision_pos_3d_scene_prop  # Bsz x N token x 6
        pos_embed_3d = self.pos_3d_proj(vision_embeds_3d_pos)
        vision_embeds_3d = vision_embeds_3d + pos_embed_3d

        # convert mask from b x n to b x q x n
        mask = mask.unsqueeze(1).repeat(1, self.args['num_query_rsp_3d'], 1).unsqueeze(1).repeat(1, self.args['num_heads_rsp_3d'], 1, 1)
        vision_embeds_3d = self.resampler_3d(vision_embeds_3d, mask)
        vision_embeds_3d = self.llama_proj_3d(vision_embeds_3d)
        return vision_embeds_3d

    def prompt_wrap(
        self, 
        img_embeds, 
        input_ids, 
        target_ids, 
        attention_mask, 
        vision_mask,
        use_system,
        vision_type,
        task_type, 
    ):
        """
        input_ids, target_ids, attention_mask: bsz x s2
        """
        input_ids = input_ids.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2
        vision_mask = vision_mask.to(self.device)

        batch_size = img_embeds.shape[0]

        # return list of headers if multiple tasks
        p_before = make_prompt_start(
            use_system=use_system, 
            vision_type=vision_type, 
            task_type=task_type,
            template=self.conv_template
        )
        if isinstance(p_before, list):
            p_before_tokens = [
                self.llama_tokenizer(p, return_tensors="pt", add_special_tokens=False)
                .input_ids[0]
                .to(self.device)
                for p in p_before
            ]
            # TODO: test in batch
            p_before_token_ids = rnn.pad_sequence(
                p_before_tokens,
                batch_first=True,
                padding_value=self.llama_tokenizer.pad_token_id,
            )  # bsz x s1
            p_before_attn_mask = p_before_token_ids.ne(
                self.llama_tokenizer.pad_token_id
            )
        else:
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False
            ).to(
                self.device
            )  # [s1, s1...] list of batch size
            p_before_token_ids = p_before_tokens.input_ids.expand(
                batch_size, -1
            )  # bsz x s1
            p_before_attn_mask = p_before_tokens.attention_mask.expand(
                batch_size, -1
            )  # bsz x s1
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(
            p_before_token_ids
        )  # .expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(
            batch_size, -1, -1
        )  # bsz x s2 x embed_dim
        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_token_ids.dtype,
                device=p_before_token_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(
            bos
        )  # bsz x 1 x embed_dim
        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1
        )  # bsz x (1+s1+NumToken+s2) x embed_dim
        # make target ids for prefix part
        empty_targets = (
            torch.ones(
                [batch_size, 1 + p_before_embeds.size()[1] + self.num_vision_token],
                dtype=torch.long,
            )
            .to(self.device)
            .fill_(-100)  # 1 (bos) + s1 + num_image_tokens (image vector)
        )  # bsz x (1 + s1 + num_image_tokens)
        targets = torch.cat(
            [empty_targets, target_ids], dim=1
        )  # bsz x (1 + s1 + num_image_tokens + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        atts_bos = torch.ones([batch_size, 1], dtype=torch.long).to(
            self.device
        )  # bsz x 1
        attention_mask = torch.cat(
            [atts_bos, p_before_attn_mask, vision_mask, attention_mask], dim=1
        )
        assert (
            attention_mask.size() == targets.size()
        )  # bsz x (1 + s1 + num_image_tokens + s2)

        return inputs_embeds, targets, attention_mask

    @torch.no_grad()
    def reconstruct_gt_input(self, gt_inputs, task_types):
        gt_inputs = copy.deepcopy(gt_inputs)

        for idx, (gt_input, task_type) in enumerate(zip(gt_inputs, task_types)):
            if task_type == 'detection':
                question, answer = gt_input[0]['value'], gt_input[1]
                bboxes = answer['value']['bboxes']
                classes = answer['value']['clses']
                index = torch.randperm(len(bboxes))

                new_answer = []
                for box_id in index:
                    template = np.random.choice(DET_ANSWER_TEMPLATE)
                    box_str = f'{str([round(x, 2) for x in bboxes[box_id].tolist()])}'
                    new_answer.append(template.format(P=box_str, C=classes[box_id]))
                new_answer = ' '.join(new_answer)

                gt_inputs[idx][0]['value'] = question
                gt_inputs[idx][1]['value'] = new_answer

        return gt_inputs

    def prepare_prompt_embeds(
        self, 
        vision_embeds, 
        vision_mask, 
        output_texts,
        vision_type, 
        task_type
    ):
        output_texts = self.reconstruct_gt_input(output_texts, task_type)
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, 
            output_texts, 
            self.max_tgt_len, 
            vision_type, 
            self.conv_template
        )
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            vision_embeds,
            input_ids,
            target_ids,
            attention_mask,
            vision_mask,
            self.use_system,
            vision_type,
            task_type,
        )
        return inputs_embeds, targets, attention_mask

    def moe_set_gate(self, input_texts, device):
        if self.training:
            input_tokens = []
            for input_text in input_texts:
                assert input_text[0]['from'] == 'human'
                token = self.llama_tokenizer(input_text[0]['value'], add_special_tokens=False).input_ids
                input_tokens.append(torch.LongTensor(token))

            input_tokens = rnn.pad_sequence(
                input_tokens, batch_first=True, padding_value=self.llama_tokenizer.pad_token_id).to(device)
            input_embeds = self.llama_model.model.model.embed_tokens(input_tokens)
        else:
            input_embeds = input_texts

        soft_gate = self.gating_network(input_embeds, reduce_token=True)
        for _, module in self.llama_model.named_modules():
            if isinstance(module, MoeLoraLayer):
                module.set_gate(soft_gate)
        return

    def get_acc(self, logits, targets):
        chosen_tokens = torch.max(logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(
            torch.long
        )  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return gen_acc

    def get_loss(self, logits, targets):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        # Flatten the tokens for model parallelism
        shift_logits = shift_logits.view(-1, self.llama_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        # AR loss
        ar_loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = ar_loss_fct(shift_logits, shift_labels)
        return loss.mean()

    def forward(self, inputs):
        task_types = []
        output_texts = []
        vision_embeds = []
        vision_masks = []
        vision_types = []

        if inputs['image'] is not None:
            image_inputs = inputs['image']
            task_types += image_inputs['task_type']
            output_texts += image_inputs['output_texts']

            image_paths = image_inputs['vision_paths']
            images = self.load_and_transform_image_data_clip(
                image_paths, self.device
            ).to(self.llama_model.dtype)  # bsz x 3 x 224 x 224
            image_embeds = self.encode_image(images)
            vision_embeds.append(image_embeds)
            vision_mask = torch.ones(
                image_embeds.shape[0], image_embeds.shape[1], device=image_embeds.device, dtype=torch.long)
            vision_masks.append(vision_mask)
            vision_types.extend(['image'] * image_embeds.shape[0])

        if inputs['pcl'] is not None:
            pcl_inputs = inputs['pcl']
            task_types += pcl_inputs['task_type']
            output_texts += pcl_inputs['output_texts']

            pcl_embeds_ = self.encode_pcl(pcl_inputs)
            # pad zero to align with image modality
            pcl_embeds = torch.zeros(
                pcl_embeds_.shape[0], 
                self.num_vision_token, 
                pcl_embeds_.shape[2],
                device=pcl_embeds_.device,
                dtype=pcl_embeds_.dtype)
            pcl_embeds[:, :self.args['num_query_rsp_3d']] = pcl_embeds_
            vision_embeds.append(pcl_embeds) 
            vision_mask = torch.ones(
                pcl_embeds.shape[0], self.num_vision_token, device=pcl_embeds.device, dtype=torch.long)
            vision_mask[:, self.args['num_query_rsp_3d']:] = 0
            vision_masks.append(vision_mask)
            vision_types.extend(['pcl'] * pcl_embeds.shape[0])

        vision_embeds = torch.cat(vision_embeds, dim=0)
        vision_masks = torch.cat(vision_masks, dim=0)
        
        inputs_embeds, targets, attention_mask = self.prepare_prompt_embeds(
            vision_embeds, vision_masks, output_texts, vision_types, task_types
        )

        self.moe_set_gate(
            output_texts,
            inputs_embeds.device,
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=None,
        )
        logits = outputs.logits
        gen_acc = self.get_acc(logits, targets)
        loss = self.get_loss(logits, targets)
        return loss, gen_acc

    # ==============================================
    # inference and evaluation 
    # ==============================================
    def transform_vision_data(self, images, device):
        image_ouputs = []
        for img in images:
            data_transform = transforms.Compose(
                [
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
            image = data_transform(img).to(device)
            image = image.half()
            image_ouputs.append(image)
        return torch.stack(image_ouputs, dim=0)

    def extract_multimodal_feature(self, inputs):
        if 'images' in inputs and inputs["images"]:  # image objects input in testing
            self.vision_type = "image"
            images = self.transform_vision_data(inputs['images'], self.device)
            image_embeds = self.encode_image(images)
            return image_embeds 
        
        if 'image_paths' in inputs and inputs["image_paths"]:
            self.vision_type = "image"
            image_paths = inputs['image_paths']
            images = self.load_and_transform_image_data_clip(
                image_paths, self.device
            ).to(self.llama_model.dtype)  # bsz x 3 x 224 x 224
            image_embeds = self.encode_image(images)
            return image_embeds

        features = []
        self.vision_type = "pcl"
        pcl_embeds = self.encode_pcl(inputs)
        features.append(pcl_embeds)
        feature_embeds = (
            torch.cat(features).sum(dim=0).unsqueeze(0)
        )  # sum all modality features together
        return feature_embeds

    def prepare_generation_embedding(self, inputs):
        """prepare for generation

        :param class inputs: model
        :return Dict: generation input
        """
        # TODO: add System header & image token size
        prompt_list = inputs["prompt"]  # questions from user
        if len(inputs["modality_embeds"]) == 1:
            feature_embeds = inputs["modality_embeds"][0]
        else:
            feature_embeds = self.extract_multimodal_feature(inputs)
            inputs["modality_embeds"].append(feature_embeds)
        
        eov = VISION_TAGS["eov"][self.vision_type]
        batch_size = feature_embeds.shape[0]
        p_before = make_prompt_start(
            use_system=False,
            vision_type=[self.vision_type for _ in range(batch_size)],
            task_type=['normal' for _ in range(batch_size)],
            template=self.conv_template
        )  # no system header in test
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(
            p_before_tokens.input_ids
        ).expand(
            batch_size, -1, -1
        )  # bsz x s1 x embed_dim

        p_after_texts = [
            f"{eov} " + prompt + f"\n{self.conv_template.sep} {self.conv_template.roles[1]}:" 
            for prompt in prompt_list
        ]
        p_after_tokens = self.llama_tokenizer(
            p_after_texts, 
            padding="longest", return_length=True, # padding right
            add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        p_after_masks_len = p_after_tokens.length.max() - p_after_tokens.length
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_tokens.input_ids.dtype,
                device=p_before_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(
            bos
        )  # bsz x 1 x embed_dim

        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, feature_embeds, p_after_embeds], dim=1
        )  # bsz x (1+s1+NumVisionToken+s2) x embed_dim

        # p_after_embeds are on right, so the pads are right, 
        # we need to move all inputs_embeds to right,
        # to make the pads on left
        tokens_len = inputs_embeds.shape[1] - p_after_masks_len
        new_inputs_embeds = torch.zeros_like(inputs_embeds)
        inputs_embeds_masks = torch.zeros(inputs_embeds.shape[:-1], 
                                         dtype=torch.int64, device=self.device)
        for idx in range(batch_size):
            inputs_embeds_masks[idx, -tokens_len[idx]:] = 1
            new_inputs_embeds[idx, -tokens_len[idx]:, :] = inputs_embeds[idx, :tokens_len[idx], :]
            new_inputs_embeds[idx, :-tokens_len[idx], :] = inputs_embeds[idx, tokens_len[idx]:, :]
        
        return inputs_embeds, inputs_embeds_masks, p_after_embeds

    def generate(self, inputs):
        """
        inputs = {
            'image_paths': optional,
            'mode': generation mode,
            'prompt': human input prompt,
            'max_tgt_len': generation length,
            'top_p': top_p,
            'temperature': temperature
            'modality_embeds': None or torch.tensor
            'modality_cache': save the image cache
        }
        """
        input_embeds, input_masks, prompt_embeds = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList(
            [LAMMStoppingCriteria([[2277, 29937], [835], [1, 2]], input_embeds)]
        )
        self.moe_set_gate(prompt_embeds, input_embeds.device)

        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            max_new_tokens=inputs["max_tgt_len"],
            top_p=inputs["top_p"],
            temperature=inputs["temperature"],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.llama_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return output_text
