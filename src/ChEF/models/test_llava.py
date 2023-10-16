import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel, StoppingCriteria
from .llava import LlavaMPTForCausalLM, LlavaLlamaForCausalLM, conv_templates, SeparatorStyle
from .utils import get_image
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .test_base import TestBase

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


def load_model(model_path, model_name, dtype=torch.float16, device='cpu', vis_processor_path = None):
    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if 'llava' in model_name.lower():
        if 'mpt' in model_name.lower():
            model = LlavaMPTForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    elif 'mpt' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)

    # get image processor
    image_processor = None
    if 'llava' in model_name.lower():
        pretrained_model_name_or_path = vis_processor_path if vis_processor_path is not None else model.config.mm_vision_tower
        image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, torch_dtype=dtype)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.get_model().vision_tower[0]
        if vision_tower.device.type == 'meta':
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device=device)
            model.get_model().vision_tower[0] = vision_tower
        else:
            vision_tower.to(device=device, dtype=dtype)
        
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    model.to(device=device)

    return tokenizer, model, image_processor, context_len

class TestLLaVA(TestBase):
    def __init__(self, model_path, vis_processor_path = None, **kwargs):
        model_name = get_model_name(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model(model_path, model_name, vis_processor_path=vis_processor_path)
        self.conv = get_conv(model_name)
        self.image_process_mode = "Resize" # Crop, Resize, Pad
        self.move_to_device()
        self.model.eval()
        
    def get_images_input_ids(self, images, prompts, dtype=torch.float16, keep_aspect_ratio=False):
        if keep_aspect_ratio:
            new_images = []
            for image, prompt in zip(images, prompts):
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = self.image_processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
                new_images.append(image.to(self.model.device, dtype=dtype))
                # replace the image token with the image patch token in the prompt (each occurrence)
                cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token, 1)
            images = new_images
        else:
            images = self.image_processor(images, return_tensors='pt')['pixel_values']
            images = images.to(self.model.device, dtype=dtype)
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256    # HACK: 256 is the max image token length hacked
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompts = [prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token) for prompt in prompts]

        input_ids = self.tokenizer(prompts).input_ids
        batch_size = len(input_ids)
        max_prompt_size = max([len(input_id) for input_id in input_ids])
        for i in range(len(input_ids)):
            padding_size = max_prompt_size - len(input_ids[i])
            input_ids[i] = [self.tokenizer.pad_token_id] * padding_size + input_ids[i]

        return images, input_ids, batch_size

    def move_to_device(self):
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        vision_tower = self.model.get_model().vision_tower[0]
        vision_tower.to(device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)
  
    @torch.no_grad()
    def do_generate(self, images, questions, dtype=torch.float16, temperature=0.2, max_new_tokens=256, stop_str=None, keep_aspect_ratio=False):
        # import ipdb;ipdb.set_trace()
        images, input_ids, batch_size = self.get_images_input_ids(images, questions, dtype, keep_aspect_ratio)
        #
        stop_idx = None
        if stop_str is not None:
            stop_idx = self.tokenizer(stop_str).input_ids
            if len(stop_idx) == 1:
                stop_idx = stop_idx[0]
            else:
                stop_idx = None
        
        output_ids = []
        get_result = [False for _ in range(batch_size)]

        input_ids = torch.as_tensor(input_ids).to(self.model.device)
        att_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        for i in range(max_new_tokens):
            if i == 0:
                # import ipdb;ipdb.set_trace()
                out = self.model(
                    input_ids,
                    attention_mask = att_mask,
                    use_cache=True,
                    images=images)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                att_mask = torch.concat([att_mask, torch.ones(batch_size, 1, device=self.model.device)], dim = -1)
                out = self.model(input_ids=token,
                            use_cache=True,
                            # attention_mask=torch.ones(batch_size, past_key_values[0][0].shape[-2] + 1, device=self.model.device),
                            attention_mask = att_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[:, -1]
            if temperature < 1e-4:
                token = torch.argmax(last_token_logits, dim=-1)
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            token = token.long().to(self.model.device)

            output_ids.append(token)
            for idx in range(len(token)):
                if token[idx] == stop_idx or token[idx] == self.tokenizer.eos_token_id:
                    get_result[idx] = True
            if all(get_result):
                break
        
        output_ids = torch.cat(output_ids, dim=1).long()
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if stop_str is not None:
            for i in range(len(outputs)):
                pos = outputs[i].rfind(stop_str)
                if pos != -1:
                    outputs[i] = outputs[i][:pos]
        return outputs

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        image = get_image(image)
        conv = self.conv.copy()
        text = question + '\n<image>'
        text = (text, image, self.image_process_mode)
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        output = self.do_generate([image], [prompt], stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)[0]

        return output
        

    @torch.no_grad()    
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        images, prompts = [], []
        for idx, (image, question) in enumerate(zip(image_list, question_list)):
            image = get_image(image)
            conv = self.conv.copy()
            text = question + '\n<image>'
            text = (text, image, self.image_process_mode)
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            images.append(image)
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        outputs = self.do_generate(images, prompts, stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)
        return outputs
    
    @torch.no_grad()
    def ppl_inference(self, image_list, question_list, answer_list, answer_pool, CoT_list = None, calib = False):
        images, prompts = [], []
        for idx, (image, question, answer) in enumerate(zip(image_list, question_list, answer_list)):
            image = get_image(image)
            conv = self.conv.copy()
            text = question + '\n<image>'
            text = (text, image, self.image_process_mode)
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            if CoT_list is not None:
                prompt += CoT_list[idx]
            prompt += '\n' + answer
            prompts.append(prompt)
            images.append(image)
        results = self.do_ppl(images, prompts, answer_list, answer_pool, calib = calib)
        return results
    
    @torch.no_grad()
    def do_ppl(self, images, prompts, answer_list, answer_pool, calib = False):
        answer_start_indices = []
        answer_end_indices = []
        template_token_list = []
        answer_token_list = []
        for template, option in zip(answer_list, answer_pool):
            template_token = self.tokenizer.encode(template, add_special_tokens=False)
            template_token_list.append(template_token)
            option_token = self.tokenizer.encode(option, add_special_tokens=False)
            if template_token != option_token:
                option_token = self.tokenizer.encode(' ' + option, add_special_tokens = False) # llava tokenizer encodes " cat" different from "cat"
            token_len = len(option_token)
            for index in range(len(template_token)):
                if template_token[index: index + token_len] == option_token:
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token)
                    break

            assert len(answer_start_indices) == len(template_token_list), "tokenizer encode answer in template different from answer only"

        images, input_ids, batch_size = self.get_images_input_ids(images, prompts, dtype=torch.float16)
        input_ids = torch.as_tensor(input_ids).to(self.model.device)
        att_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        outputs = self.model.transformer(input_ids = input_ids, images = images, attention_mask = att_mask)
        logits = F.linear(outputs.last_hidden_state, self.model.transformer.wte.weight)
        logits = logits[:,:-1]
        input_ids = input_ids[:,1:]
        start_indices, end_indices = [], []
        for i in range(len(answer_list)):
            token_len = len(template_token_list[i])
            for index in range(input_ids.shape[1]-token_len, 0, -1):
                if input_ids[i,index: index+token_len].cpu().numpy().tolist() == template_token_list[i]:
                    start_indices.append(index + answer_start_indices[i])
                    end_indices.append(index + answer_end_indices[i])
                    input_ids[i,:index] = -100
                    break
            assert len(start_indices) == (i+1), "tokenizer encode answer different from answer in conversation"

        results = []
        if calib:
            for idx, item_logits in enumerate(logits):
                score = 0.0
                item_prob = F.softmax(item_logits[start_indices[idx]:end_indices[idx]], dim=-1)
                for jdx in range(end_indices[idx]-start_indices[idx]):
                    score += torch.log(item_prob[jdx, answer_token_list[idx][jdx]]).item()
                score = score/len(answer_token_list[idx])
                results.append(score)
        else:
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), input_ids.reshape(-1),ignore_index=-100, reduction='none')
            loss = loss.reshape(-1, input_ids.shape[1]).float()
            for idx, item_loss in enumerate(loss):
                results.append(item_loss[start_indices[idx]:end_indices[idx]].mean().item())
        return results
    
    def get_icl_prompt(self, question, conv, ice, incontext_cfg):  # TODO: unified between models
        if incontext_cfg['add_sysmsg']:
            conv.system += incontext_cfg['sysmsg']
        if incontext_cfg['use_pic']:
            raise NotImplementedError
        else:
            icl_question = ''
            for j in range(incontext_cfg['ice_num']):
                if not isinstance(ice[j]['gt_answers'], list):
                    icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers']}. "
                else:
                    icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers'][0]}. "
            icl_question = icl_question + question + '\n<image>'
        return icl_question

    @torch.no_grad()
    def icl_batch_generate(self, image_list, question_list, ices, incontext_cfg, max_new_tokens=128):
        images, prompts = [], []
        for image, question, ice in zip(image_list, question_list, ices):
            image = get_image(image)
            conv = self.conv.copy()
            text = self.get_icl_prompt(question, conv, ice, incontext_cfg)
            text = (text, image, self.image_process_mode)
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            images.append(image)
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        outputs = self.do_generate(images, prompts, stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)
        return outputs, prompts
    
    @torch.no_grad()
    def do_calibration(self, image_list, question_list, answer_list, answer_pool, CoT_list=None):
        return super().do_calibration(image_list, question_list, answer_list, answer_pool, CoT_list)


    @torch.no_grad()
    def icl_ppl_inference(self, image_list, question_list, answer_list, answer_pool, ices, incontext_cfg, CoT_list = None):
        images, prompts = [], []
        for idx, (image, question, ice, answer) in enumerate(zip(image_list, question_list, ices, answer_list)):
            image = get_image(image)
            conv = self.conv.copy()
            text = self.get_icl_prompt(question, conv, ice, incontext_cfg)
            text = (text, image, self.image_process_mode)
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            if CoT_list is not None:
                prompt += CoT_list[idx]
            prompt += '\n' + answer
            prompts.append(prompt)
            images.append(image)
        results = self.do_ppl(images, prompts, answer_list, answer_pool)
        return results, prompts
