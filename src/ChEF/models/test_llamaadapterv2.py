import torch
from . import llama_adapter_v2
from .utils import *
import torch.nn.functional as F
from .test_base import TestBase

CONV_VISION = Conversation(
    system='Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n',
    roles=("Instruction", "Input", "Response"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="### ",
)

class TestLLamaAdapterV2(TestBase):
    def __init__(self, model_path, 
                    max_seq_len = 1024,
                    max_batch_size = 40,
                    **kwargs) -> None:
        llama_dir = model_path
        model, preprocess = llama_adapter_v2.load("LORA-BIAS-7B", llama_dir, download_root=llama_dir, max_seq_len=max_seq_len, max_batch_size=max_batch_size)
        self.img_transform = preprocess
        self.model = model.eval()
        self.tokenizer = self.model.tokenizer
        self.move_to_device()

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256):
        imgs = [get_BGR_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama_adapter_v2.format_prompt(question)]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        result = results[0].strip()
        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        imgs = [get_BGR_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama_adapter_v2.format_prompt(question) for question in question_list]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        results = [result.strip() for result in results]
        return results

    @torch.no_grad()
    def do_ppl(self, images, prompts, answer_list, answer_pool, calib = False):
        answer_start_indices = []
        answer_end_indices = []
        template_token_list = []
        answer_token_list = []
        for idx, (template, option) in enumerate(zip(answer_list, answer_pool)):
            template_token = self.tokenizer.encode('Response:' + template,  bos=False, eos=False)[2:]
            template_token_list.append(template_token)
            if template == option:
                option_token = self.tokenizer.encode('Response:' + option,  bos=False, eos=False)[2:]
            else: 
                option_token = self.tokenizer.encode(option,  bos=False, eos=False)
            token_len = len(option_token)
            for index in range(len(template_token)):
                if template_token[index: index + token_len] == option_token:
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token)
                    break
            assert len(answer_start_indices) == len(template_token_list), "tokenizer encode answer in template different from answer only"

        logits, target_ids = self.model.ppl_generate(images, prompts, answer_list)
        
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
        imgs = [get_BGR_image(image) for image in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama_adapter_v2.format_prompt(question) for question in question_list]
        if CoT_list is not None:
            prompts = [prompt + ' ' + cot + '\n' for prompt, cot in zip(prompts, CoT_list)]
        results = self.do_ppl(imgs, prompts, answer_list, answer_pool, calib=calib)
        return results
        
    def get_icl_prompt(self, question_list, chat_list, ices, incontext_cfg):
        prompts =[]
        for question, conv, ice in zip(question_list, chat_list, ices):
            if incontext_cfg['add_sysmsg']:
                conv.system += incontext_cfg['sysmsg']
                conv.system += '\n\n'
            conv.append_message(conv.roles[0], '\n')
            if incontext_cfg['use_pic']:
                raise NotImplementedError
            else:
                icl_question = ''
                for j in range(incontext_cfg['ice_num']):
                    if not isinstance(ice[j]['gt_answers'], list):
                        icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers']}. "
                    else:
                        icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers'][0]}. "
                icl_question += f"{question}: \n\n"
                conv.messages[-1][-1] += icl_question
            
            conv.append_message(conv.roles[2], '')
            prompt = conv.get_prompt()
            prompts.append(prompt)
        return prompts
    
    @torch.no_grad()
    def icl_batch_generate(self, image_list, question_list, ices, incontext_cfg, max_new_tokens=256):
        imgs = [get_BGR_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        chat_list = [CONV_VISION.copy() for _ in range(len(question_list))]
        prompts = self.get_icl_prompt(question_list, chat_list, ices, incontext_cfg)
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        results = [result.strip() for result in results]
        return results, prompts
    
    @torch.no_grad()
    def icl_ppl_inference(self, image_list, question_list, answer_list, answer_pool, ices, incontext_cfg, CoT_list = None):
        imgs = [get_BGR_image(image) for image in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        chat_list = [CONV_VISION.copy() for _ in range(len(question_list))]
        prompts = self.get_icl_prompt(question_list, chat_list, ices, incontext_cfg)
        if CoT_list is not None:
            prompts = [prompt + ' ' + cot + '\n' for prompt, cot in zip(prompts, CoT_list)]
        results = self.do_ppl(imgs, prompts, answer_list, answer_pool)
        return results, prompts
    
    def do_calibration(self, image_list, question_list, answer_list, answer_pool, CoT_list=None):
        return super().do_calibration(image_list, question_list, answer_list, answer_pool, CoT_list)
        