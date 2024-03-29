import torch
import torch.nn.functional as F

from .test_base import TestBase
from model.llava.model.builder import load_pretrained_model, get_conv
from model.llava.model.language_model.llava_llama import LlamaForCausalLM
from model.llava.mm_utils import (
    get_model_name_from_path, 
    tokenizer_image_token,
    KeywordsStoppingCriteria
)
from model.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
)
from model.llava.conversation import SeparatorStyle


def batch_tokenizer_image_token(prompts: list, tokenizer, add_special_tokens=True):
    input_ids_list = []
    for prompt in prompts:
        input_ids = tokenizer_image_token(prompt, tokenizer, 
            IMAGE_TOKEN_INDEX, return_tensors='pt', add_special_tokens=add_special_tokens)
        input_ids_list.append(input_ids)

    input_ids_list = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id if add_special_tokens else IGNORE_INDEX)
    return input_ids_list

class TestLLaVA15(TestBase):
    def __init__(self, model_path, model_base = None, vis_processor_path = None, device='cuda', **kwargs):
        model_name = get_model_name_from_path(model_path)
        print(f'Load model on device map: {device}')
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(model_path, model_base, model_name, vis_processor_path=vis_processor_path, device=device)
        self.conv = get_conv(model_name)
        self.stop_str = [self.conv.sep if self.conv.sep_style!=SeparatorStyle.TWO else self.conv.sep2]
        self.image_process_mode = "Resize" # Crop, Resize, Pad
        self.model.eval()
        self.dtype = self.model.dtype
        self.device = self.model.device
    
    def build_conversation(
        self, 
        idx,
        image_list, 
        prompt, 
        CoT_answer_list=None, 
        batch_answers=None,
        **kwargs,
    ):    
        conv = self.conv.copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)    
        prompt = conv.get_prompt()
        if CoT_answer_list is not None:
            prompt += CoT_answer_list[idx]
        if batch_answers is not None:
            prompt += '\n' + batch_answers[idx]
        return prompt
    
    def build_input_image(self, image_list):
        image_list = self.get_image_list(image_list)
        image_tensors = []
        for image in image_list:
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(self.device)
            image_tensors.append(image_tensor)
        image_tensors = torch.cat(image_tensors)
        return image_tensors

    @torch.no_grad()
    def do_generate(self, 
        images, 
        prompt, 
        max_new_tokens=30,
        **kwargs):
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stopping_criteria = KeywordsStoppingCriteria(self.stop_str, self.tokenizer, input_ids)
        input_token_len = input_ids.shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[images],
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        output = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_token=True)[0]
        output = output.strip()
        for idx in range(len(self.stop_str[0])):
            if output.endswith(self.stop_str[0][:idx+1]):
                output = output[:-(idx+1)]
                break
        output = output.strip()
        return output
            
    @torch.no_grad()
    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        batch_input_ids = batch_tokenizer_image_token(batch_prompt, self.tokenizer).to(self.device)
        batch_option_ids = batch_tokenizer_image_token(batch_options, self.tokenizer, add_special_tokens=False).to(self.device)
        (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = \
            self.model.prepare_inputs_labels_for_multimodal(
            batch_input_ids, None, None, None, batch_input_ids, batch_images)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
        )

        logits = outputs.logits
        logits = logits[:, :-1].float()
        labels = labels[:, 1:]
        results = []
        for idx in range(labels.shape[0]):
            option_len = torch.sum(batch_option_ids[idx]!=IGNORE_INDEX).item() 
            non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
            start_index = non_zero_indices.max() - option_len + 1
            end_index = start_index + option_len
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean().item()
            results.append(score)
        return results
