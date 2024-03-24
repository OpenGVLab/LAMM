import torch
import torch.nn.functional as F
import numpy as np
from .shikra.builder.build_shikra import load_pretrained_shikra
from .shikra import model_args, training_args
from .shikra.dataset.process_function import PlainBoxFormatter
from .shikra.dataset.builder import prepare_interactive, SingleImageInteractive
from .test_base import TestBase

class TestShikra(TestBase):
    def __init__(self, 
                 model_path,
                 device='cuda',
                 encoder_ckpt_path = None, 
                 **kwargs
                 ):
        model_args.model_name_or_path = model_path
        if encoder_ckpt_path is not None:
            model_args.vision_tower = encoder_ckpt_path
        training_args.device=device
        model, self.preprocessor = load_pretrained_shikra(model_args, training_args)
        model.to(dtype=torch.float16, device=device)
        model.model.vision_tower.to(dtype=torch.float16, device=device)
        self.model = model
        self.preprocessor['target'] = {'boxes': PlainBoxFormatter()}
        self.tokenizer = self.preprocessor['text']
        self.tokenizer.padding_side = 'left'
        self.gen_kwargs = dict(
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        self.gen_kwargs['do_sample'] = False
        self.device = device

    def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None, batch_answers=None, **kwargs):
        ds = prepare_interactive(model_args, self.preprocessor)
        ds.set_image(self.build_input_image(image_list))
        ds.append_message(role=ds.roles[0], message = prompt, boxes = [], boxes_seq = [])
        user_msg = ''
        if CoT_answer_list is not None:
            user_msg += CoT_answer_list[idx]
        if batch_answers is not None:
            user_msg += '\n ' + batch_answers[idx]
        if user_msg != '':
            ds.append_message(role=ds.roles[1], message = user_msg, boxes = [], boxes_seq = [])
        return ds
    

    def build_input_image(self, image_list):
        image_list = self.get_image_list(image_list)
        if len(image_list) == 1:
            image = image_list[0]
        else:
            # as shikra don't support multiimage input, we concat in horizon
            image = self.horizontal_concat(image_list) 
        return image
    
    def do_generate(self, image_list: list, ds: SingleImageInteractive, max_new_tokens, **kwargs):
        model_inputs = ds.to_model_input()
        text = model_inputs['input_text']
        images = model_inputs['images'].to(dtype=torch.float16, device=self.device)
        input_dict = self.tokenizer(
            [text], 
            padding="longest", return_length=True, 
            add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        output_ids = self.model.generate(
            images = images,
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            **self.gen_kwargs
        )
        input_token_len = input_ids.shape[-1]
        output = self.tokenizer.batch_decode(output_ids[:, input_token_len:])[0]
        return output.split('</s>')[0]


    def do_ppl(self, batch_images, ds_list, batch_options, **kwargs):
        text, images = [], []
        for ds in ds_list:
            model_inputs = ds.to_model_input()
            text.append(model_inputs['input_text'])
            images.append(model_inputs['images'].to(dtype=torch.float16, device=self.device))
        images = torch.cat(images, dim = 0)
        input_dict = self.tokenizer(
            text, 
            padding="longest", return_length=True, 
            add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        option_dict = self.tokenizer(
            batch_options,
            padding="longest", return_length=True, 
            add_special_tokens=False, return_tensors="pt"
        )
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        option_ids = option_dict['input_ids']
       
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
        )
        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        logits = logits[:, :-1].float()
        labels = input_ids[:, 1:]
        
        results = []
        for idx in range(labels.shape[0]):
            option_len = torch.sum(option_ids[idx]!=0).item() 
            end_index = len(labels[idx]) - 1
            start_index = end_index - option_len
            if not np.all(labels[idx][start_index: end_index].detach().cpu().numpy() == option_ids[idx][-option_len:].numpy()):
                import ipdb;ipdb.set_trace()
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, option_ids[idx][-option_len:]]).mean().item()
            results.append(score)
        return results