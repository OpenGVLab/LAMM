import torch
import torch.nn.functional as F

from .instruct_blip.models import load_model_and_preprocess
from .utils import get_image
from .test_base import TestBase

class TestInstructBLIP(TestBase):
    def __init__(self, device) -> None:
        self.device = device#torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device)
        self.tokenizer = self.model.llm_tokenizer
        self.model.max_txt_len = 512

    def build_conversation(self, 
        idx,
        image_list, 
        prompt, 
        CoT_answer_list=None, 
        batch_answers=None,
        **kwargs,):
        if CoT_answer_list is not None:
            prompt += CoT_answer_list[idx]
        if batch_answers is not None:
            prompt += '\n' + batch_answers[idx]
        return prompt 

    def build_input_image(self, image_list):
        image_list = self.get_image_list(image_list)
        image_tensors = []
        for image in image_list:
            image_tensor = self.vis_processors['eval'](image).unsqueeze(0).to(self.device)
            image_tensors.append(image_tensor)
            image_tensors.append(image_tensor)
        image_tensors = torch.cat(image_tensors)
        image_tensors = image_tensors.permute(1, 0, 2, 3)
        return image_tensors

    def do_generate(self, imgs, prompt, max_new_tokens, **kwargs):
        imgs = imgs.unsqueeze(0)
        output = self.model.generate({"image": imgs, "prompt": prompt}, max_length=max_new_tokens)[0]
        return output

    def do_ppl(self, batch_images, batch_prompt, batch_answers, batch_options, calib=None, **kwargs):
        answer_start_indices = []
        answer_end_indices = []
        answer_token_list = []
        for template, option in zip(batch_answers, batch_options):
            template_token = self.tokenizer.encode(template, add_special_tokens=False)
            option_token = self.tokenizer.encode(option, add_special_tokens=False)
            token_len = len(option_token)
            for index in range(len(template_token)):
                if template_token[index: index + token_len] == option_token:
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token)
                    break
        output, target_ids = self.model.forward_multiple({"image": batch_images, "text_input": batch_prompt, "text_output": batch_answers})
        logits = output['logits'][:,:-1]
        target_ids = target_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), target_ids.reshape(-1),ignore_index=-100, reduction='none')
        loss = loss.reshape(-1, target_ids.shape[1]).float()
        target_ids = target_ids.cpu()
        mask = target_ids != -100
        indices = torch.argmax(mask.long().to(target_ids) * torch.arange(target_ids.size(1),0, -1), dim=-1)
        indices[mask.sum(dim=1) == 0] = -1
        start_indices = indices.tolist()
        end_indices = indices.tolist()
        for i in range(len(batch_answers)):
            start_indices[i] = start_indices[i] + answer_start_indices[i]
            end_indices[i] = end_indices[i] + answer_end_indices[i]
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
            for idx, item_loss in enumerate(loss):
                results.append(item_loss[start_indices[idx]: end_indices[idx]].mean().item())
        return results

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128, **kwargs):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question}, max_length=max_new_tokens)[0]

        return output
    
    # @torch.no_grad()
    # def batch_generate(self, image_list, question_list, max_new_tokens=128, **kwargs):
    #     imgs = [get_image(img) for img in image_list]
    #     imgs = [self.vis_processors["eval"](x) for x in imgs]
    #     imgs = torch.stack(imgs, dim=0).to(self.device)
    #     output = self.model.generate({"image": imgs, "prompt": question_list}, max_length=max_new_tokens)
    #     return output

    # @torch.no_grad()
    # def do_ppl(self, images, prompts, answer_list, answer_options, calib = False):
    #     answer_start_indices = []
    #     answer_end_indices = []
    #     answer_token_list = []
    #     # import pdb
    #     # pdb.set_trace()
    #     for template, option in zip(answer_list, answer_options):
    #         template_token = self.tokenizer.encode(template, add_special_tokens=False)
    #         option_token = self.tokenizer.encode(option, add_special_tokens=False)
    #         token_len = len(option_token)
    #         for index in range(len(template_token)):
    #             if template_token[index: index + token_len] == option_token:
    #                 answer_start_indices.append(index)
    #                 answer_end_indices.append(index + token_len)
    #                 answer_token_list.append(option_token)
    #                 break
    #     output, target_ids = self.model.forward({"image": images, "text_input": prompts, "text_output": answer_list})
    #     logits = output['logits'][:,:-1]
    #     target_ids = target_ids[:, 1:]
    #     loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), target_ids.reshape(-1),ignore_index=-100, reduction='none')
    #     loss = loss.reshape(-1, target_ids.shape[1]).float()
    #     target_ids = target_ids.cpu()
    #     mask = target_ids != -100
    #     indices = torch.argmax(mask.long().to(target_ids) * torch.arange(target_ids.size(1),0, -1), dim=-1)
    #     indices[mask.sum(dim=1) == 0] = -1
    #     start_indices = indices.tolist()
    #     end_indices = indices.tolist()
    #     for i in range(len(answer_list)):
    #         start_indices[i] = start_indices[i] + answer_start_indices[i]
    #         end_indices[i] = end_indices[i] + answer_end_indices[i]
    #     results = []
    #     if calib:
    #         for idx, item_logits in enumerate(logits):
    #             score = 0.0
    #             item_prob = F.softmax(item_logits[start_indices[idx]:end_indices[idx]], dim=-1)
    #             for jdx in range(end_indices[idx]-start_indices[idx]):
    #                 score += torch.log(item_prob[jdx, answer_token_list[idx][jdx]]).item()
    #             score = score/len(answer_token_list[idx])
    #             results.append(score)
    #     else:
    #         for idx, item_loss in enumerate(loss):
    #             results.append(item_loss[start_indices[idx]: end_indices[idx]].mean().item())
    #     return results
        

    @torch.no_grad()
    def ppl_inference(self, batch_images, batch_prompt, batch_options, batch_answers, CoT_list = None, calib = False, **kwargs):
        input_images, input_prompts = [], []
        for idx, (image_list, prompt) in \
                enumerate(zip(batch_images, batch_prompt)):
            input_prompt = self.build_conversation(idx, image_list, prompt, **kwargs)
            input_image_list = self.build_input_image(image_list)
            input_prompts.append(input_prompt)
            input_images.append(input_image_list)
        input_images = torch.stack(input_images, dim=0).to(self.device)
        prompts = batch_prompt
        if CoT_list is not None:
            prompts = [prompt + '\n' + cot for prompt, cot in zip(batch_prompt, CoT_list)]
        results = self.do_ppl(input_images, prompts, batch_answers, batch_options, calib = calib)
        return results
    