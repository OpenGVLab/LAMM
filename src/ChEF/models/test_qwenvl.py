import torch
import torch.nn.functional as F
import numpy as np
from .test_base import TestBase
from .qwen import (
    QWenConfig,
    QWenTokenizer,
    QWenLMHeadModel,
    make_context,
    get_stop_words_ids,
    decode_tokens
)

class TestQwenVL(TestBase):
    def __init__(self, model_path, device='cuda', **kwargs):
        self.model_config = QWenConfig.from_pretrained(model_path)
        self.model = QWenLMHeadModel.from_pretrained(model_path, device_map=device)
        self.tokenizer = QWenTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.stop_words_ids = get_stop_words_ids(
            self.model.generation_config.chat_format, self.tokenizer
        )

    def build_conversation(
        self, 
        idx, 
        image_list,
        prompt, 
        CoT_answer_list=None, 
        batch_answers=None, 
        **kwargs):
        if isinstance(image_list, str):
            image_list = [image_list]
        format_list = []
        for image in image_list:
            format_list.append(dict(image=image))
        format_list.append(dict(text=prompt))
        query = self.tokenizer.from_list_format(format_list)
        raw_text, context_tokens = make_context(self.tokenizer, \
            query=query, system="You are a helpful assistant.")
        if CoT_answer_list is not None:
            raw_text += CoT_answer_list[idx]
            context_tokens += self.tokenizer.encode(CoT_answer_list[idx], 
                allowed_special=set(self.tokenizer.IMAGE_ST))
        if batch_answers is not None:
            raw_text += '\n ' + batch_answers[idx]
            context_tokens += self.tokenizer.encode("\n") + self.tokenizer.encode(' ' + batch_answers[idx], 
                allowed_special=set(self.tokenizer.IMAGE_ST))
        return (raw_text, context_tokens)
    
    def build_input_image(self, image_list):
        return None
    
    def do_generate(self, image_list, prompt: tuple, max_new_tokens, **kwargs):
        raw_text, context_tokens = prompt
        input_ids = torch.tensor([context_tokens]).to(self.device)
        outputs = self.model.generate(
            input_ids,
            stop_words_ids=self.stop_words_ids,
            return_dict_in_generate=False,
            generation_config=self.model.generation_config,
            do_sample=False,
            **kwargs,
        )

        response = decode_tokens(
            outputs[0],
            self.tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=self.model.generation_config.chat_format,
            verbose=False,
            errors='replace'
        )
        return response
    
    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        raw_text, context_tokens = ([item[0] for item in batch_prompt], [torch.tensor(item[1]) for item in batch_prompt])
       
        input_ids = torch.nn.utils.rnn.pad_sequence(
            context_tokens,
            batch_first=True,
            padding_value=0.).to(self.device)
        attention_mask = input_ids.ne(0.)
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = transformer_outputs[0]

        logits = self.model.lm_head(hidden_states)
        logits = logits[:, :-1]
        labels = input_ids
        labels = labels[:, 1:]
        
        results = []
        batch_option_ids = []
        for option in batch_options:
            batch_option_ids.append(self.tokenizer.encode(f' {option}', add_special_tokens=False, return_tensors='pt').squeeze(0))
        for idx in range(labels.shape[0]):
            # qwen encodes "!" as 0
            option_len = len(batch_option_ids[idx])
            non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
            start_index = non_zero_indices.max() - option_len + 1
            end_index = start_index + option_len
            last_zero_num = option_len - torch.nonzero(batch_option_ids[idx], as_tuple=False).squeeze().max() - 1 
            if last_zero_num > 0:
                start_index += last_zero_num
                end_index += last_zero_num
            if not np.all(labels[idx][start_index: end_index].detach().cpu().numpy() == batch_option_ids[idx].numpy()):
                import ipdb;ipdb.set_trace()
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean().item()
            results.append(score)
        return results