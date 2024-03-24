import torch
import torch.nn.functional as F
import numpy as np
from .test_base import TestBase
from .rlhfv import (
    init_muffin, 
    wrap_question_with_default_conv, 
    torch_pad_sequence,
    KeywordsStoppingCriteria,
)

class TestRLHFV(TestBase):
    def __init__(self, model_path, device='cuda', **kwargs):
        model, image_processor, image_token_len, tokenizer = init_muffin(model_path=model_path, device=device)
        self.model = model
        self.image_processor = image_processor
        self.image_token_len = image_token_len
        self.tokenizer = tokenizer
        self.model.eval()
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.keywords = ['###']

    def build_conversation(
        self, 
        idx,
        image_list, 
        prompt, 
        CoT_answer_list=None, 
        batch_answers=None,
        **kwargs,
    ):    
        if isinstance(image_list, str):
            image_list = [image_list]
        prompt = wrap_question_with_default_conv(prompt, self.image_token_len * len(image_list))
        if CoT_answer_list is not None:
            prompt += CoT_answer_list[idx]
        if batch_answers is not None:
            prompt += ' ' + batch_answers[idx]
        return prompt
    
    def build_input_image(self, image_list):
        image_list = self.get_image_list(image_list)
        image_tensors = []
        for image in image_list:
            image_tensor = self.image_processor(image).half().to(self.device)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors)
        return image_tensors
    
    def do_generate(self, images, prompt, max_new_tokens, **kwargs):
        
        tokenized = self.tokenizer([prompt])

        input_ids = [torch.as_tensor(v) for v in tokenized['input_ids']]
        input_ids = torch_pad_sequence(input_ids, self.tokenizer.pad_token_id, padding_side='left')
        input_size = input_ids.shape[-1]
        attn_mask = [torch.as_tensor(v) for v in tokenized['attention_mask']]
        attn_mask = torch_pad_sequence(attn_mask, 0, padding_side='left')
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_size)
        output = self.model.generate(
            input_ids=input_ids.to(self.device),
            images=[images],
            attention_mask=attn_mask.to(self.device),
            temperature=0.7,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=[stopping_criteria],
            repetition_penalty=1.1)

        output_id = output.sequences[0]
        response = self.tokenizer.decode(output_id[input_size:], skip_special_tokens=True)
        if response.count('###'):
            response = response[: response.index('###')]
        response = response.strip()
        return response
    
    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        tokenized = self.tokenizer(batch_prompt)

        input_ids = [torch.as_tensor(v) for v in tokenized['input_ids']]
        input_ids = torch_pad_sequence(input_ids, self.tokenizer.pad_token_id, padding_side='left')
        input_size = input_ids.shape[-1]
        attn_mask = [torch.as_tensor(v) for v in tokenized['attention_mask']]
        attn_mask = torch_pad_sequence(attn_mask, 0, padding_side='left')
        
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            images=batch_images,
            attention_mask=attn_mask.to(self.device),
            labels=input_ids,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
        )

        logits = outputs.logits
        logits = logits[:, :-1].float()
        labels = input_ids
        labels = labels[:, 1:]
        
        results = []
        batch_option_ids = []
        for option in batch_options:
            batch_option_ids.append(self.tokenizer.encode(f'{option}', return_tensors='pt', add_special_tokens=False).squeeze(0))
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