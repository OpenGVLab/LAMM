import logging
import io
from petrel_client.client import Client
import numpy as np
import os
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image, ImageFile
import requests
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from transformers import LlamaForCausalLM, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList

from .CLIP import load as load_clip
import model.LAMM.conversations as conversations
from .modeling_llama import LlamaForCausalLM
from .utils.pcl_utils import MEAN_COLOR_RGB, random_sampling
from .utils.data import transform_vision_data

# load optional 3d encoder
try:
    LOAD_EPCL_EXT = True
    from .EPCL import build_epcl_encoder
except ImportError as e:
    LOAD_EPCL_EXT = False
    logging.warning(f'{e.msg}. Please refer to README.md to install optional extension for 3D environment if required.')

# load optional lightllm
try:
    LOAD_LIGHTLLM_EXT = True
    from .modeling_lightllm import LlamaLightForCausalLM
except ImportError as e:
    LOAD_LIGHTLLM_EXT = False
    logging.warning(f'{e.msg}. Please refer to README.md to install optional LightLLM extension if required.')

ImageFile.LOAD_TRUNCATED_IMAGES = True


VISION_TAGS = {
    "pos": {"image": "<image>", "pcl": "<pcl>"},
    "sov": {"image": "<Img>", "pcl": "<Pcl>"},
    "eov": {"image": "</Img>", "pcl": "</Pcl>"},
}


class LAMMStoppingCriteria(StoppingCriteria):
    def __init__(self, stops, input_ids, device):
        """intialize stopping criteria

        :param list stops: list of stop tokens
        :param list input_ids: input ids
        """
        super().__init__()
        self.stops = [torch.tensor(stop).to(device) for stop in stops]
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


def build_one_instance(tokenizer, conversation, vision_type="image", template=conversations.default_conversation):
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
            text = f"{eov} " + turn["value"] + "\n{} {}: ".format(template.sep, template.roles[1])
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(
                one_input_id
            )  # do not perform loss regression on human prompt
        else:
            if role == "human":
                # text = "{}: ".format(template.roles[0]) + turn["value"] + "\n### {}:".format(template.roles[1])
                text = "{}: {}\n{} {}: ".format(template.roles[0], turn["value"], template.sep, template.roles[1])
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100] * len(one_input_id)
            elif role == "gpt":
                text = turn["value"] + "\n{}".format(template.sep2 if (template.sep2 is not None) else template.sep)
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception(f"{role} is a Wrong Role!!!")
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids


def process_batch_instance(
    tokenizer, batch_of_conversations, max_tgt_len, vision_type="image", template=conversations.default_conversation
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
            tokenizer, conversation, vision_type=vision_type, template=template
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


def make_prompt_start(use_system=False, vision_type="image", task_type="normal", template=conversations.default_conversation):
    """make starting prompt

    :param bool use_system: whether to use system message, defaults to False
    :param str vision_type: type of visio data, defaults to 'image'
    :param str task_type: task type of current sample, defaults to 'normal'
    :return str: resulting starting prompt
    """
    # PROMPT_START = f'### Human: {VISION_TAGS["sov"][vision_type]}'
    PROMPT_START = f'{template.sep} {template.roles[0]}: {VISION_TAGS["sov"][vision_type]}'
    if use_system:
        if task_type == "normal":
            # print(template.system)
            return f"{template.system}\n\n" + PROMPT_START
        else:
            if template.sys_temp is None:
                return [
                    f"{conversations.conversation_dict[task]}\n\n" + PROMPT_START
                    for task in task_type
                ]
            else:
                # print(template.sys_temp.format(system_message=conversations.conversation_dict[task_type[0]]))
                return [template.sys_temp.format(system_message=conversations.conversation_dict[task]) + PROMPT_START for task in task_type]
    else:
        return PROMPT_START


class LAMMPEFTModel(nn.Module):
    """LoRA for LAMM model"""

    def __init__(self, **args):
        super(LAMMPEFTModel, self).__init__()
        self.args = args
        self.client = None

        self.vision_type = args["vision_type"] if "vision_type" in args else "image"
        encoder_pretrain = (
            args["encoder_pretrain"] if "encoder_pretrain" in args else "clip"
        )
        assert encoder_pretrain in [
            "clip",
            "epcl",
        ], f"Encoder_pretrain: {encoder_pretrain} Not Implemented"
        encoder_ckpt_path = (
            args["encoder_ckpt_path"]
            if not encoder_pretrain == "clip"
            else "~/.cache/clip/ViT-L-14.pt"
        )
        llm_ckpt_path = args["llm_ckpt_path"]
        use_system = args["use_system"] if "use_system" in args else False
        self.conv_template = conversations.conv_templates[args['conv_template']] if 'conv_template' in args else conversations.default_conversation
        self.stage = args["stage"]

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
            if LOAD_EPCL_EXT is False:
                raise ImportError('Please refer to README.md to install extension for 3D environment.')

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
        print(f"Initializing language decoder from {llm_ckpt_path} ...")
        self.initialize_language_model(llm_ckpt_path)
        print("Language decoder initialized.")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            llm_ckpt_path, use_fast=False
        )
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        tokens = self.get_special_tokens()
        self.add_tokens(tokens)
        self.build_projection_layer()

        self.max_tgt_len = args["max_tgt_len"]
        self.use_system = use_system
        self.use_flash_attn = args.get('use_flash_attn', False)
        self.use_xformers = args.get('use_xformers', False)
        self.device = torch.cuda.current_device()
        
    
    def initialize_language_model(self, llm_ckpt_path):
        # add the lora module
        peft_config = self.build_peft_config()

        if self.args.get('use_lightllm', False):
            if LOAD_LIGHTLLM_EXT is False:
                raise ImportError('Please refer to README.md to install LightLLM extension.')

            self.llama_model = LlamaLightForCausalLM(
                batch_size=self.args['bs'],
                max_input_len=1024,
                max_output_len=self.args['max_tgt_len'],
                weight_dir=llm_ckpt_path,
                lora_path=self.args['delta_ckpt_path'],
                lora_config=peft_config,
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(llm_ckpt_path)
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()

    def init_client(self):
        if self.client is not None:
            return
        self.client = Client("~/petreloss.conf")

    def build_projection_layer(self):
        self.llama_proj = nn.Linear(
            self.vision_hidden_size, self.llama_model.config.hidden_size
        )
        print("LLaMa projection layer initialized.")

    def build_peft_config(self):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=self.args["lora_target_modules"],
        )
        return peft_config

    def get_special_tokens(self):
        tokens = []
        return tokens
    
    def add_tokens(self, tokens):
        if len(tokens) == 0:
            return 
        
        # Add an empty token to match len(tokenizer) == len(model.input_embeddings)
        self.llama_tokenizer.add_tokens(['<XSFQ/>'], special_tokens=True)
        # Add special tokens
        num_new_tokens = self.llama_tokenizer.add_tokens(tokens, special_tokens=True)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if num_new_tokens > 0:
            input_embeddings = self.llama_model.get_input_embeddings().weight.data
            output_embeddings = self.llama_model.get_output_embeddings().weight.data

            input_embedding_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embedding_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            input_embeddings[-num_new_tokens:] = input_embedding_avg
            output_embeddings[-num_new_tokens:] = output_embedding_avg

    def encode_image(self, image_paths):
        """encode images to llama inputs

        :param tupe image_paths: (bsz, )
        :return tensor, tensor: input feature to llama, attention mask to llama
        """
        if self.encoder_pretrain == "clip":
            inputs = self.load_and_transform_image_data_clip(
                image_paths, self.device
            )  # bsz x 3 x 224 x 224
            inputs = inputs.to(dtype=self.llama_model.dtype, device=self.device)  # clip requires torch.float32
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
        inputs = inputs.to(dtype=self.llama_model.dtype)  # clip requires torch.float32
    
        if self.vision_feature_type == "global":
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)  # bsz x 768
            image_embeds = embeddings.to(self.llama_model.dtype)
            inputs_llama = self.llama_proj(image_embeds).unsqueeze(
                1
            )  # bsz x 1 x llama_size
        elif self.vision_feature_type == "local":
            with torch.no_grad():
                embeddings = self.visual_encoder.forward_patch_features(inputs.to(self.device))[
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
        self.init_client()
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
            elif 'gqa' in image_path:
                idx = image_path.find('gqa')
                image_path = f'gqa:s3://mmg_gqa/train_images{image_path[idx+10:]}'
                img_bytes = self.client.get(image_path)
                assert img_bytes is not None, f'data path {image_path} is invalid'
                byte_stream = io.BytesIO(img_bytes)
                image = Image.open(byte_stream).convert('RGB')
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

    def embed_tokens(self, token_ids):
        # peft model need deeper call
        return self.llama_model.model.model.embed_tokens(token_ids)

    def prompt_wrap(
        self, img_embeds, input_ids, target_ids, attention_mask, use_system, task_type
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
            use_system=use_system, vision_type=self.vision_type, task_type=task_type, template=self.conv_template
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
        p_before_embeds = self.embed_tokens(p_before_token_ids) # .expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        p_after_embeds = self.embed_tokens(input_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_token_ids.dtype,
                device=p_before_token_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.embed_tokens(bos)  # bsz x 1 x embed_dim
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
        ), "{} expected but {} given".format(self.valid_type, inputs["vision_type"])
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
            self.llama_tokenizer, output_texts, self.max_tgt_len, self.vision_type, self.conv_template
        )
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            vision_embeds,
            input_ids,
            target_ids,
            attention_mask,
            self.use_system,
            task_type,
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=not self.use_flash_attn,
        )
        loss = outputs.loss
        # calculate the token accuarcy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(
            torch.long
        )  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    def ppl_forward(self, inputs):
        assert (
            self.vision_type == inputs["vision_type"]
        ), "{} expected but {} given".format(self.valid_type, inputs["vision_type"])
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
            self.llama_tokenizer, output_texts, self.max_tgt_len, self.vision_type, self.conv_template
        )
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            vision_embeds,
            input_ids,
            target_ids,
            attention_mask,
            self.use_system,
            task_type,
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=not self.use_flash_attn,
        )
        logits = outputs.logits
        return logits, targets

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

        batch_size = feature_embeds.shape[0]
        p_before = make_prompt_start(
            vision_type=self.vision_type, template=self.conv_template
        )  # no system header in test
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        p_before_embeds = self.embed_tokens(
            p_before_tokens.input_ids
        ).expand(
            batch_size, -1, -1
        )  # bsz x s1 x embed_dim

        p_after_texts = [f"{eov} " + prompt + f"\n{self.conv_template.sep} {self.conv_template.roles[1]}:" for prompt in prompt_list]
        p_after_tokens = self.llama_tokenizer(
            p_after_texts, 
            padding="longest", return_length=True, # padding right
            add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        p_after_masks_len = p_after_tokens.length.max() - p_after_tokens.length
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)

        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_tokens.input_ids.dtype,
                device=p_before_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )  # bsz x 1
        bos_embeds = self.embed_tokens(
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
            [LAMMStoppingCriteria([[2277, 29937], [835], [1, 2]], input_embeds, input_embeds.device)]            # TODO: different template has corresponding end signal [sep2]
        )
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            max_new_tokens=inputs["max_tgt_len"],
            top_p=inputs["top_p"],
            temperature=inputs["temperature"],
            do_sample=False,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.llama_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return output_text


class LAMMSFTModel(LAMMPEFTModel):
    """SFT for LAMM model"""

    def initialize_language_model(self, llm_ckpt_path):
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_ckpt_path)
        
        # freeze language decoder
        if self.stage == 1:
            print('Freeze language decoder for stage 1 trainning')
            self.llama_model.model.requires_grad_(False)
        
        if self.stage == 2:
            self.gradient_checkpointing = self.args['gradient_checkpointing']
            # enable gradient checkpointing
            if self.gradient_checkpointing:
                print('Enable gradient checkpointing for SFT')
                self.llama_model.model.gradient_checkpointing = True
            print("Enable language decoder for stage 2 training")
            self.llama_model.model.requires_grad_(True)
        # self.llama_model.print_trainable_parameters()

    def build_projection_layer(self):
        super().build_projection_layer()
        if self.stage == 2:
            print("Load projector weights for stage 2 training")
            self.load_stage1_weights(self.args['llm_proj_path'])

    def load_stage1_weights(self, ckpt_path):
        original_state_dict = torch.load(ckpt_path)
        lm_head_weights = {}
        llama_proj_weights = {}
        for key, value in original_state_dict.items():
            if key.startswith('llama_model.lm_head'):
                lm_head_weights[key.split('.')[-1]] = value
            elif key.startswith('llama_proj'):
                llama_proj_weights[key.split('.')[-1]] = value
        self.llama_proj.load_state_dict(llama_proj_weights)
        self.llama_model.lm_head.load_state_dict(lm_head_weights)

    def embed_tokens(self, token_ids):
        return self.llama_model.model.embed_tokens(token_ids)