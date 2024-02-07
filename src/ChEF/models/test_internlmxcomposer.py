import torch
import torch.nn.functional as F
import numpy as np
from .test_base import TestBase
from .internlm import (
    InternLMXComposer2ForCausalLM, 
    InternLMXcomposer2Config, 
    InternLMXComposer2Tokenizer
)

class TestInternlmXcomposer(TestBase):
    def __init__(self, model_path, device='cuda', **kwargs):
        self.model_config = InternLMXcomposer2Config.from_pretrained(model_path)
        self.model = InternLMXComposer2ForCausalLM.from_pretrained(model_path, device_map=device)
        self.tokenizer = InternLMXComposer2Tokenizer.from_pretrained(model_path)
        self.model.tokenizer = self.tokenizer
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
        **kwargs):
        prompt = f'<ImageHere>{prompt}' # TODO: make this interleaved
        prompt = self.model.build_inputs(prompt, [])
        if CoT_answer_list is not None:
            prompt += CoT_answer_list[idx]
        if batch_answers is not None:
            prompt += '\n' + batch_answers[idx]
        return prompt
    
    def build_input_image(self, image_list):
        image_list = self.get_image_list(image_list)
        image_tensors = []
        for image in image_list:
            image_tensor = self.model.vis_processor(image).unsqueeze(0).to(self.device)
            image_tensors.append(image_tensor)
        image_tensors = torch.cat(image_tensors)
        return image_tensors
    
    def do_generate(self, image_list: list, prompt: str, max_new_tokens, **kwargs):
        image, _, _ = self.model.img2emb(image_list)
        inputs, im_mask = self.model.interleav_wrap_chat(self.tokenizer, prompt, image)
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items() if torch.is_tensor(v)
        }
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
        ]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=0.8,
            eos_token_id=eos_token_id,
            repetition_penalty=1.005,
            im_mask=im_mask,
            **kwargs,
        )
        outputs = outputs[0].cpu().tolist()
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split('[UNUSED_TOKEN_145]')[0]
        return response
    
    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        to_regress_embeds, attention_mask, targets, im_mask, token_ids = self.model.interleav_wrap(batch_images, batch_prompt)
        im_mask = im_mask.bool()
        outputs = self.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=to_regress_embeds,
            im_mask=im_mask,
        )
        hidden_states = outputs[0]
        logits = self.model.output(hidden_states)
        logits = logits.float()
        logits = logits[:, :-1]
        labels = token_ids[:, 1:]
        results = []
        batch_option_ids = []
        for option in batch_options:
            batch_option_ids.append(self.tokenizer.encode(f' {option}', add_special_tokens=False, return_tensors='pt').squeeze(0))
        for idx in range(labels.shape[0]):
            option_len = len(batch_option_ids[idx])
            non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
            start_index = non_zero_indices.max() - option_len + 1
            end_index = start_index + option_len
            if not np.all(labels[idx][start_index: end_index].detach().cpu().numpy() == batch_option_ids[idx].numpy()):
                import ipdb;ipdb.set_trace()
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean().item()
            results.append(score)
        return results