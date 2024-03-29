import torch
import torch.nn.functional as F
import numpy as np
from .instruct_blip.models import load_model_and_preprocess
from .test_base import TestBase

class TestInstructBLIP(TestBase):
    def __init__(self, device) -> None:
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device)
        self.tokenizer = self.model.llm_tokenizer
        self.model.max_txt_len = 512

    def build_conversation(self, 
        idx,
        image_list, 
        prompt, 
        CoT_answer_list=None, 
        batch_answers=None,
        **kwargs):
        if CoT_answer_list is not None:
            prompt += '\n' + CoT_answer_list[idx]
        # we add answers in text_output
        # if batch_answers is not None:
        #     prompt += '\n' + batch_answers[idx]
        return prompt 

    def build_input_image(self, image_list):
        image_list = self.get_image_list(image_list)
        image_tensors = []
        for image in image_list:
            image_tensor = self.vis_processors['eval'](image).unsqueeze(0).to(self.device)
            image_tensors.append(image_tensor)
        image_tensors = torch.cat(image_tensors)
        image_tensors = image_tensors.permute(1, 0, 2, 3)
        return image_tensors

    def do_generate(self, imgs, prompt, max_new_tokens, **kwargs):
        imgs = imgs.unsqueeze(0)
        output = self.model.generate({"image": imgs, "prompt": prompt}, max_length=max_new_tokens)[0]
        return output

    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        batch_answers = kwargs['batch_answers']
        batch_images = torch.stack(batch_images, dim=0).to(self.device)
        batch_option_ids = []
        for option in batch_options:
            batch_option_ids.append(self.tokenizer.encode(f'{option}', add_special_tokens=False, return_tensors='pt').squeeze(0)) 
        output, labels = self.model.forward_multiple(
            {"image": batch_images, "text_input": batch_prompt, "text_output": batch_answers})
        logits = output['logits'][:,:-1].float()
        labels = labels[:, 1:]
        results = []
        for idx in range(labels.shape[0]):
            option_len = len(batch_option_ids[idx])
            non_zero_indices = torch.nonzero(labels[idx]!=-100, as_tuple=False).squeeze()
            start_index = non_zero_indices.max() - option_len
            end_index = start_index + option_len
            if not np.all(labels[idx][start_index: end_index].detach().cpu().numpy() == batch_option_ids[idx].numpy()):
                import ipdb;ipdb.set_trace()
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean().item()
            results.append(score)
        return results
    
    