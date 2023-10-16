import torch
from .instruct_blip.models import load_model_and_preprocess
from .utils import get_image
import torch.nn.functional as F

class TestInstructBLIP:
    def __init__(self) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device)
        self.tokenizer = self.model.llm_tokenizer
        self.model.max_txt_len = 512

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question}, max_length=max_new_tokens)[0]

        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        output = self.model.generate({"image": imgs, "prompt": question_list}, max_length=max_new_tokens)
        return output

    @torch.no_grad()
    def do_ppl(self, images, prompts, answer_list, answer_options, calib = False):
        answer_start_indices = []
        answer_end_indices = []
        answer_token_list = []
        for template, option in zip(answer_list, answer_options):
            template_token = self.tokenizer.encode(template, add_special_tokens=False)
            option_token = self.tokenizer.encode(option, add_special_tokens=False)
            token_len = len(option_token)
            for index in range(len(template_token)):
                if template_token[index: index + token_len] == option_token:
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token)
                    break
        output, target_ids = self.model.forward({"image": images, "text_input": prompts, "text_output": answer_list})
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
        for i in range(len(answer_list)):
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
    def ppl_inference(self, image_list, question_list, answer_list, answer_pool, CoT_list = None, calib = False):
        imgs = [get_image(image) for image in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = question_list
        if CoT_list is not None:
            prompts = [prompt + '\n' + cot for prompt, cot in zip(question_list, CoT_list)]
        results = self.do_ppl(imgs, prompts, answer_list, answer_pool, calib = calib)
        return results
    

    def get_icl_prompt(self, question_list, ices, incontext_cfg):
        prompts =[]
        for question, ice in zip(question_list, ices):
            icl_question = ''
            if incontext_cfg['add_sysmsg']:
                icl_question += incontext_cfg['sysmsg']
            if incontext_cfg['use_pic']:
                raise NotImplementedError
            else:
                for j in range(incontext_cfg['ice_num']):
                    if not isinstance(ice[j]['gt_answers'], list):
                        icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers']}. "
                    else:
                        icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers'][0]}. "
                icl_question += f"{question}: "
            prompts.append(icl_question)

        return prompts
    
    @torch.no_grad()
    def icl_batch_generate(self, image_list, question_list, ices, incontext_cfg, max_new_tokens=128):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = self.get_icl_prompt(question_list, ices, incontext_cfg)
        output = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        return output, prompts
    
    @torch.no_grad()
    def icl_ppl_inference(self, image_list, question_list, answer_list, answer_options, ices, incontext_cfg, CoT_list = None, calib = False):
        imgs = [get_image(image) for image in image_list]
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = self.get_icl_prompt(question_list, ices, incontext_cfg)
        if CoT_list is not None:
            prompts = [prompt + '\n' + cot for prompt, cot in zip(prompts, CoT_list)]
        results = self.do_ppl(imgs, prompts, answer_list, answer_options, calib = calib)
        return results, prompts