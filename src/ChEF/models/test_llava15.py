import os
import warnings
import shutil
from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, StoppingCriteria, CLIPImageProcessor
import torch
from model.llava.model import *
from model.llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX

import torch.nn.functional as F

from .llava import conv_templates, SeparatorStyle
from .utils import get_image
from .test_base import TestBase
from model.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from transformers.generation.streamers import TextIteratorStreamer

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False
    
def get_model_name(model_path):
    # get model name
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        model_name = model_paths[-2] + "_" + model_paths[-1]
    else:
        model_name = model_paths[-1]
    
    return model_name


def get_conv(model_name):
    if "llava" in model_name.lower():
        if "v1" in model_name.lower():
            template_name = "llava_v1"
        elif "mpt" in model_name.lower():
            template_name = "mpt_multimodal"
        else:
            template_name = "multimodal"
    elif "mpt" in model_name:
        template_name = "mpt_text"
    elif "koala" in model_name: # Hardcode the condition
        template_name = "bair_v1"
    elif "v1" in model_name:    # vicuna v1_1/v1_2
        template_name = "vicuna_v1_1"
    else:
        template_name = "v1"
    return conv_templates[template_name].copy()


def load_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", dtype=torch.float16, vis_processor_path = None):
    # import ipdb; ipdb.set_trace()
    if not torch.cuda.is_available():
        dtype = torch.float32
        device = 'cpu'
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
           vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor
        
        '''
        pretrained_model_name_or_path = vis_processor_path if vis_processor_path is not None else model.config.mm_vision_tower
        image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, torch_dtype=dtype)
        # image_processor.to(device=device, dtype=dtype)
        '''
        
        model.to(device=device, dtype=dtype)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


# def batch_tokenizer_image_token(prompts: list, tokenizer): TODO
#     input_ids_list = []
#     for prompt in prompts:
#         input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
#         input_ids_list.append(input_ids)
    
#     max_prompt_size = max([len(input_ids) for input_ids in input_ids_list])
#     for i in range(len(input_ids_list)):
#         padding_size = max_prompt_size - len(input_ids_list[i])
#         input_ids_list[i] = [tokenizer.pad_token_id] * padding_size + input_ids_list[i]
    
#     return torch.tensor(input_ids_list).cuda()

class TestLLaVA15(TestBase):
    def __init__(self, model_path, model_base = None, vis_processor_path = None, **kwargs):
        model_name = get_model_name(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model(model_path, model_base, model_name, vis_processor_path=vis_processor_path)
        self.conv = get_conv(model_name)
        self.image_process_mode = "Resize" # Crop, Resize, Pad
        # self.move_to_device()
        self.model.eval()
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
    
    @torch.no_grad()
    def do_generate(self, image_tensor, prompt, stop_str=None,dtype=torch.float16, temperature=0.2, top_p=1.0, max_new_tokens=30, keep_aspect_ratio=False, do_sample=False):
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)

        # print(f"do_sample: {do_sample}, temperature: {temperature}, top_p: {top_p}, max_new_tokens: {max_new_tokens}" )
        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=dict(
                inputs=input_ids,
                images=image_tensor,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]))
            thread.start()
            # workaround: second-to-last token is always " "
            # but we want to keep it if it's not the second-to-last token
            prepend_space = False
            output = ""
            for new_text in streamer:
                if new_text == " ":
                    prepend_space = True
                    continue
                if new_text.endswith(stop_str):
                    new_text = new_text[:-len(stop_str)].strip()
                    prepend_space = False
                elif prepend_space:
                    new_text = " " + new_text
                    prepend_space = False
                if len(new_text):
                    output+=new_text
                    # yield new_text
            if prepend_space:
                output+=" "
                # yield " "
            thread.join()
        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, prompt_list, **kwargs):
        outputs = []
        for idx, (image, question) in enumerate(zip(image_list, prompt_list)):
            image_data = get_image(str(image))
            conv = self.conv.copy()
            image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)    
            prompt = conv.get_prompt()
            # import ipdb; ipdb.set_trace()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            output=self.do_generate(image_tensor, prompt, stop_str, self.dtype)
            outputs.append(output)
        return outputs
            
    @torch.no_grad()
    def ppl_inference(self, image_list, question_list, answer_list, answer_pool, CoT_list = None, calib = False, **kwargs):
        images, prompts = [], []
        results = []
        for idx, (image, question, answer) in enumerate(zip(image_list, question_list, answer_list)):
            image_data = get_image(str(image))
            image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
            conv = self.conv.copy()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            if CoT_list is not None:
                prompt += CoT_list[idx]
            prompt += '\n' + answer
            result = self.do_ppl(image_tensor, prompt, answer, answer_pool[idx])
            results.append(result)
        return results
    
    @torch.no_grad()
    def do_ppl(self, image_tensor, prompt, template, option):
        
        option_token = self.tokenizer.encode(option, add_special_tokens=False)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        outputs = self.model(input_ids = input_ids, images = image_tensor)
        logits = outputs.logits
        logits = logits[:,:-1]
        input_ids = input_ids[:,1:]

        # token_len = len(template_token)
        # for index in range(input_ids.shape[1]-token_len, 0, -1):
        #     if input_ids[0 ,index: index+token_len].cpu().numpy().tolist() == template_token:
        #         start_indice = index + answer_start_indice
        #         end_indice = index + answer_end_indice
        #         input_ids[:,:index] = -100
        #         break
        # assert start_indice is not None, "tokenizer encode answer different from answer in conversation"
        input_ids = torch.tensor(option_token)[1:].cuda()
        logits = logits[:, -len(input_ids):]


        loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), input_ids.reshape(-1),ignore_index=-100, reduction='none')
        loss = loss.mean().item()
        return loss
