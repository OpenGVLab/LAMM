import os
import torch
from .test_llava15 import (
    TestLLaVA15, 
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IMAGE_PATCH_TOKEN, 
    get_conv,
    SeparatorStyle
)
from src.model.llava.model import LlavaLlamaForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer

class TestLLaVARLHF(TestLLaVA15):
    def __init__(self, model_path, device=None, **kwargs):
        sft_path = os.path.join(model_path, 'sft_model')
        model = LlavaLlamaForCausalLM.from_pretrained(
            sft_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        lora_path = os.path.join(model_path, 'rlhf_lora_adapter_model')
        model = PeftModel.from_pretrained(
            model,
            lora_path,
        )
        self.model = model
        self.model.eval()
        
        self.conv = get_conv("llava-llava_v1")
        self.stop_str = [self.conv.sep if self.conv.sep_style!=SeparatorStyle.TWO else self.conv.sep2]
        self.image_process_mode = "Resize" # Crop, Resize, Pad

        self.move_to_device(device)
        self.model.device = device
        tokenizer = AutoTokenizer.from_pretrained(sft_path, use_fast=False)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.model.device, dtype=self.model.dtype)
        self.image_processor = vision_tower.image_processor