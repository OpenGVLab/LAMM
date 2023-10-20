import io

import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.nn.utils import rnn


from .utils.conversations import default_conversation, conversation_dict
from .utils.header import *
from transformers import StoppingCriteria, StoppingCriteriaList

from .CLIP import load as load_clip

from .modeling_llama import LlamaForCausalLM
from .utils.pcl_utils import MEAN_COLOR_RGB, RandomCuboid, random_sampling
from .utils.data import transform_vision_data

ImageFile.LOAD_TRUNCATED_IMAGES = True


VISION_TAGS = {
    "pos": {"image": "<image>", "pcl": "<pcl>"},
    "sov": {"image": "<Img>", "pcl": "<Pcl>"},
    "eov": {"image": "</Img>", "pcl": "</Pcl>"},
}


class LAMMStoppingCriteria(StoppingCriteria):
    def __init__(self, stops, input_ids):
        """intialize stopping criteria

        :param list stops: list of stop tokens
        :param list input_ids: input ids
        """
        super().__init__()
        self.stops = [torch.tensor(stop).to('cuda') for stop in stops]
        self.stop_flag = [0] * input_ids.shape[0]

    def check_stop(self, input_ids):
        """check whether to stop generation

        :param list input_ids: input token ids
        :return bool: stop or not
        """
        for stop in self.stops:
            if torch.all((stop == input_ids[-len(stop):])).item():
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """call function of stop creteria

        :param torch.LongTensor output_ids: output token ids
        :return bool: stop or not
        """
        flag = 1
        for id, output_id in enumerate(output_ids):
            if self.stop_flag[id] == 1:
                continue
            if self.check_stop(output_id):
                self.stop_flag[id] = 1
            else:
                flag = 0
        if flag == 1:
            return True
        return False


def build_one_instance(tokenizer, conversation, vision_type="image"):
    """build one instance for training; text part

    :param class tokenizer: text tokenizer
    :param list conversation: list of conversation
    :param str vision_type: type of vision data, defaults to 'image'
    :raises Exception: Exception if wrong role included
    :return list: conversation text list, input token ids, target token ids
    """
    pos = VISION_TAGS["pos"][vision_type]
    eov = VISION_TAGS["eov"][vision_type]

    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn["from"]
        if i == 0:  # the first human turn
            assert role == "human"
            turn["value"] = (
                turn["value"].replace(f"{pos}\n", "").replace(f"\n{pos}", "")
            )
            text = f"{eov} " + turn["value"] + "\n### Assistant:"
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(
                one_input_id
            )  # do not perform loss regression on human prompt
        else:
            if role == "human":
                text = "Human: " + turn["value"] + "\n### Assistant:"
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100] * len(one_input_id)
            elif role == "gpt":
                text = turn["value"] + "\n###"
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception("Wrong Role!!!")
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids


def process_batch_instance(
    tokenizer, batch_of_conversations, max_tgt_len, vision_type="image"
):
    """build one batch of instance for training

    :param class tokenizer: text tokenizer
    :param list batch_of_conversations: batch of conversations
    :param int max_tgt_len: max token length of after vision tokens
    :param str vision_type: type of vision data, defaults to 'image'
    :return list: input token ids, target token ids, attention mask
    """
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(
            tokenizer, conversation, vision_type=vision_type
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


def make_prompt_start(use_system=False, vision_type="image", task_type="normal", icl_sysmsg=None):
    """make starting prompt

    :param bool use_system: whether to use system message, defaults to False
    :param str vision_type: type of visio data, defaults to 'image'
    :param str task_type: task type of current sample, defaults to 'normal'
    :return str: resulting starting prompt
    """
    PROMPT_START = f'### Human: {VISION_TAGS["sov"][vision_type]}'
    
    if icl_sysmsg is not None:
        PROMPT_START = icl_sysmsg + PROMPT_START
    
    if use_system:
        if task_type == "normal":
            return f"{default_conversation.system}\n\n" + PROMPT_START
        else:
            return [
                f"{conversation_dict[task]}\n\n" + PROMPT_START
                for task in task_type
            ]
    else:
        return PROMPT_START


class LAMMPEFTModel(nn.Module):
    """LoRA for LAMM model"""

    def __init__(self, **args):
        super(LAMMPEFTModel, self).__init__()
        self.args = args

        self.vision_type = args["vision_type"] if "vision_type" in args else "image"
        encoder_pretrain = (
            args["encoder_pretrain"] if "encoder_pretrain" in args else "clip"
        )
        assert encoder_pretrain in [
            "clip",
            "epcl",
        ], f"Encoder_pretrain: {encoder_pretrain} Not Implemented"
        encoder_cache_ckpt_path = "~/.cache/clip/ViT-L-14.pt" if encoder_pretrain == 'clip' else None
        encoder_ckpt_path = args.get('encoder_ckpt_path', encoder_cache_ckpt_path)
        vicuna_ckpt_path = args["vicuna_ckpt_path"]

        use_system = args["use_system"] if "use_system" in args else False
        stage = args["stage"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(
            f"Initializing [{encoder_pretrain}] visual encoder from {encoder_ckpt_path} [{device}]..."
        )

        # -1 for last embedding; -2 for transformer output
        self.vision_feature_type = args["vision_feature_type"]
        self.num_vision_token = args["num_vision_token"]
        self.encoder_pretrain = encoder_pretrain

        # TODO: Make sure the number of vision tokens is correct
        if self.encoder_pretrain.lower() == "clip":
            clip_encoder, self.visual_preprocess = load_clip("ViT-L/14", device=device)
            self.visual_encoder = clip_encoder.visual
            if self.vision_feature_type == "global":  # global feature from CLIP
                self.vision_hidden_size = 768
                self.num_vision_token = 1
                assert self.num_vision_token == 1, "Only 1 global token is available!"
            elif self.vision_feature_type == "local":  # patch features from CLIP ViT
                self.vision_hidden_size = 1024
                self.num_vision_token = min(
                    self.num_vision_token, 256
                )  # may cut partial tokens

        elif self.encoder_pretrain.lower() == "epcl":
            # PCL data Processing
            self.use_color = (
                self.args["use_color"] if "use_color" in self.args else False
            )
            self.use_height = (
                self.args["use_height"] if "use_height" in self.args else False
            )
            self.num_points = (
                self.args["num_points"] if "num_points" in self.args else 40000
            )

            if self.vision_feature_type == "global":
                raise NotImplementedError("Global feature not implemented for EPCL")
            else:
                self.vision_hidden_size = 1024
                self.num_vision_token = self.num_vision_token
            from .EPCL import build_epcl_encoder
            self.visual_encoder = build_epcl_encoder(
                pretrain=True, store_path=encoder_ckpt_path, device=device
            )
        else:
            raise NotImplementedError(
                f"Encoder {self.encoder_pretrain} not implemented!"
            )

        # freeze vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print("Visual encoder initialized.")

        print(f"Initializing language decoder from {vicuna_ckpt_path} ...")
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=self.args["lora_target_modules"],
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            vicuna_ckpt_path, use_fast=False
        )
        self.llama_tokenizer.add_tokens(['</UNKOWN>'])
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print("Language decoder initialized.")

        self.llama_proj = nn.Linear(
            self.vision_hidden_size, self.llama_model.config.hidden_size
        )
        print("LLaMa projection layer initialized.")

        self.max_tgt_len = args["max_tgt_len"]
        self.use_system = use_system
        self.device = torch.cuda.current_device()

    def encode_image(self, image_paths):
        """encode images to llama inputs

        :param tupe image_paths: (bsz, )
        :return tensor, tensor: input feature to llama, attention mask to llama
        """
        if self.encoder_pretrain == "clip":
            inputs = self.load_and_transform_image_data_clip(
                image_paths, self.device
            )  # bsz x 3 x 224 x 224
            inputs = inputs.to(self.llama_model.dtype)  # clip requires torch.float32
            inputs_llama = self.clip_encode_image(inputs)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
                self.device
            )  # bsz x 1/256
            return inputs_llama, atts_llama
        else:
            raise NotImplementedError("Encoder not implemented!")

    def encode_image_object(self, images):
        """encoder loaded image objects"""
        if self.encoder_pretrain == "clip":
            inputs = transform_vision_data(
                images, self.device
            )  # bsz x 3 x 224 x 224
            inputs_llama = self.clip_encode_image(inputs)  # bsz x 1/256 x llama_size
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
                self.device
            )  # bsz x 1/256
            return inputs_llama, atts_llama
        else:
            raise NotImplementedError(
                "Encoder pretrain [{}] not implemented".format(self.encoder_pretrain)
            )

    def encode_pcl(self, pcl_paths):
        # load pcl data
        inputs = self.load_and_transform_pcl_data(
            pcl_paths, self.device
        )  # bsz x 40000 x 3

        inputs = inputs.to(self.llama_model.dtype)  # clip requires torch.float32
        with torch.no_grad():
            if self.vision_feature_type == "global":
                raise NotImplementedError("Global feature not implemented for pcl")
            elif self.vision_feature_type == "local":
                embeddings = self.visual_encoder(inputs)[1][
                    :, : self.num_vision_token
                ]  # bsz x 256 x 1024;
                image_embeds = embeddings.reshape(-1, self.vision_hidden_size).to(
                    self.llama_model.dtype
                )  # bsz*num vision token x 1024
        inputs_llama = self.llama_proj(image_embeds).reshape(
            -1, self.num_vision_token, self.llama_model.config.hidden_size
        )  # bsz x num_vision_token x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
            self.device
        )  # bsz x 1/256
        return inputs_llama, atts_llama

    def clip_encode_image(self, inputs):
        inputs = inputs.to(self.llama_model.dtype)  # clip requires torch.float32
    
        if self.vision_feature_type == "global":
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)  # bsz x 768
            image_embeds = embeddings.to(self.llama_model.dtype)
            inputs_llama = self.llama_proj(image_embeds).unsqueeze(
                1
            )  # bsz x 1 x llama_size
        elif self.vision_feature_type == "local":
            with torch.no_grad():
                embeddings = self.visual_encoder.forward_patch_features(inputs)[
                    :, : self.num_vision_token
                ]  # bsz x self.num_vision_token x 1024
            image_embeds = embeddings.reshape(-1, self.vision_hidden_size).to(
                self.llama_model.dtype
            )  # bsz*num vision token x 1024
            inputs_llama = self.llama_proj(image_embeds).reshape(
                -1, self.num_vision_token, self.llama_model.config.hidden_size
            )  # bsz x num_vision_token x llama_size
        else:
            raise NotImplementedError(
                "{} not Implemented".format(self.vision_feature_type)
            )
        return inputs_llama

    def load_and_transform_image_data_clip(self, image_paths, device):
        if image_paths is None:
            return None
        image_ouputs = []
        for image_path in image_paths:
            if isinstance(image_path, Image.Image):
                image = image_path
            elif os.path.exists(image_path):
                image = Image.open(image_path)
            elif image_path.startswith("http://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                print("can not load image: ", image_path)
            image_output = self.visual_preprocess(image).to(device)  # 3 x 224 x 224
            image_ouputs.append(image_output)
        return torch.stack(image_ouputs, dim=0)  # B x 3 x 224 x 224

    def load_and_transform_pcl_data(self, pcl_paths, device):
        if pcl_paths is None:
            return None
        pcl_output = []
        for pcl_path in pcl_paths:
            mesh_vertices = np.load(pcl_path)  # 150000, 3
            if not self.use_color:
                point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            else:
                point_cloud = mesh_vertices[:, 0:6]
                point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

            if self.use_height:
                floor_height = np.percentile(point_cloud[:, 2], 0.99)
                height = point_cloud[:, 2] - floor_height
                point_cloud = np.concatenate(
                    [point_cloud, np.expand_dims(height, 1)], 1
                )

            point_cloud, _ = random_sampling(
                point_cloud, self.num_points, return_choices=True
            )
            pcl_output.append(torch.from_numpy(point_cloud))
        return torch.stack(pcl_output, dim=0).to(device)  # bsz x num_points x 3

    def prompt_wrap(
        self, img_embeds, input_ids, target_ids, attention_mask, use_system, task_type, icl_sysmsg=None
    ):
        """
        input_ids, target_ids, attention_mask: bsz x s2
        """
        input_ids = input_ids.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2

        batch_size = img_embeds.shape[0]

        # return list of headers if multiple tasks
        p_before = make_prompt_start(
            use_system=use_system, vision_type=self.vision_type, task_type=task_type, icl_sysmsg=icl_sysmsg
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
        )  # bsz x (1 + s1 + 1)
        targets = torch.cat(
            [empty_targets, target_ids], dim=1
        )  # bsz x (1 + s1 + num_image_tokens + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        # atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + self.num_vision_token], dtype=torch.long).to(self.device) # bsz x (1[bos] + s1 +num_image_tokens)
        atts_bos = torch.ones([batch_size, 1], dtype=torch.long).to(
            self.device
        )  # bsz x 1
        atts_img = torch.ones([batch_size, self.num_vision_token], dtype=torch.long).to(
            self.device
        )  # bsz x num_image_tokens
        attention_mask = torch.cat(
            [atts_bos, p_before_attn_mask, atts_img, attention_mask], dim=1
        )
        assert (
            attention_mask.size() == targets.size()
        )  # bsz x (1 + s1 + num_image_tokens + s2)
        return inputs_embeds, targets, attention_mask

    def forward(self, inputs):
        # image_paths = inputs['image_paths']
        assert (
            self.vision_type == inputs["vision_type"]
        ), "{} expected but {} given".format(self.vision_type, inputs["vision_type"])
        task_type = inputs["task_type"]
        vision_paths = inputs["vision_paths"]
        if self.vision_type == "image":
            vision_embeds, _ = self.encode_image(vision_paths)
        elif self.vision_type == "pcl":
            vision_embeds, _ = self.encode_pcl(vision_paths)  # Bsz x N token x C
        else:
            raise ValueError("vision type [{}] not supported".format(self.vision_type))

        output_texts = inputs["output_texts"]
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len, self.vision_type
        )
        icl_sysmsg = inputs.get("icl_sysmsg")
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            vision_embeds,
            input_ids,
            target_ids,
            attention_mask,
            self.use_system,
            task_type,
            icl_sysmsg
        )
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        return outputs.logits, targets

    def extract_multimodal_feature(self, inputs):
        """Extract multimodal features from the input in Generation (Test)

        :param Dict inputs: input dict; modality: path
        :return _type_: _description_
        """
        features = []
        if 'image_paths' in inputs and inputs["image_paths"]:
            image_embeds, _ = self.encode_image(inputs["image_paths"])
            features.append(image_embeds)
        if 'images' in inputs and inputs["images"]:  # image objects input in testing
            image_embeds, _ = self.encode_image_object(inputs["images"])
            return image_embeds
        if "pcl_paths" in inputs and inputs["pcl_paths"]:
            pcl_embeds, _ = self.encode_pcl(inputs["pcl_paths"])
            features.append(pcl_embeds)
        # TODO: Cautions HERE! Multimodality allowed in test ONLY!
        feature_embeds = (
            torch.cat(features).sum(dim=0).unsqueeze(0)
        )  # sum all modality features together
        return feature_embeds

    def prepare_generation_embedding(self, inputs):
        """prepare for generation

        :param class inputs: model
        :return Dict: generation input
        """
        eov = VISION_TAGS["eov"][self.vision_type]
        # TODO: add System header & image token size
        prompt_list = inputs["prompt"]  # questions from user
        if len(inputs["modality_embeds"]) == 1:
            feature_embeds = inputs["modality_embeds"][0]
        else:
            feature_embeds = self.extract_multimodal_feature(inputs)
            inputs["modality_embeds"].append(feature_embeds)
        # import ipdb;ipdb.set_trace()
        batch_size = feature_embeds.shape[0]
        p_before = make_prompt_start(
            vision_type=self.vision_type
        )  # no system header in test
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(
            p_before_tokens.input_ids
        ).expand(
            batch_size, -1, -1
        )  # bsz x s1 x embed_dim

        p_after_texts = [f"{eov} " + prompt + "\n### Assistant:" for prompt in prompt_list]
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

        return new_inputs_embeds, inputs_embeds_masks

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
        input_embeds, input_masks = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList(
            [LAMMStoppingCriteria([[2277, 29937], [835]], input_embeds)]
        )
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
#self.model.llama_tokenizer.batch_decode(logits[0,-5].argsort(dim=-1)[-50:], skip_special_tokens=True)
#self.model.llama_tokenizer.batch_decode(logits[0,-7:].argmax(dim=-1), skip_special_tokens=True)
#@elf.model.llama_tokenizer.batch_decode(logits[0,-7:].argmax(dim=-1), skip_special_tokens=True)