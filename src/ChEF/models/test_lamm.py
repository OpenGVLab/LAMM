import torch
import torch.nn.functional as F
from model.LAMM.conversations import conv_templates
from .utils import get_image
from .test_base import TestBase
from model.LAMM import LAMMPEFTModel
import numpy as np
class TestLAMM(TestBase):
    def __init__(self, 
                 model_path,
                 device=None,
                 task_type = 'normal',
                 **kwargs
                 ):
        self.conv_mode = 'simple'
        self.model = LAMMPEFTModel(**kwargs)
        delta_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)             # TODO: load delta_ckpt from model_path in lamm_3d.yaml
        self.model = self.model.eval().half()
        self.task_type = task_type
        self.move_to_device(device)
        self.model.device = device

    def generate_conversation_text(self, input_list, history = [], sys_msg = None):
        """get all conversation text

        :param args args: input args
        :param str question: current input from user
        :param list history: history of conversation, [(q, a)]
        """
        conv = conv_templates[self.conv_mode]
        if sys_msg:
            conv.system = sys_msg
        prompts_list = []
        for input in input_list:
            prompts = ''
            prompts += conv.system 
            for q, a in history:
                prompts += "{} {}: {}\n{} {}: {}\n".format(conv.sep, conv.roles[0], q, conv.sep, conv.roles[1], a)
            prompts += "{} {}: {}\n".format(conv.sep, conv.roles[0], input)
            prompts_list.append(prompts)
        return prompts_list
    
    def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None, batch_answers=None, generate=True, **kwargs):
        if generate:
            prompts = self.generate_conversation_text([prompt], sys_msg=kwargs.get('sys_msg',None))
            prompt = prompts[0]
            if CoT_answer_list is not None:
                prompt += CoT_answer_list[idx]
        else:
            conversation = []
            conversation.append({'from':'human', 'value': prompt})
            fromgpt = batch_answers[idx]
            if CoT_answer_list is not None:
                fromgpt = CoT_answer_list[idx] + '\n' + fromgpt
            conversation.append({'from':'gpt', 'value': fromgpt})
            prompt = conversation
        return prompt
    
    def build_input_image(self, image_list):
        image_list = self.get_image_list(image_list)
        if len(image_list) == 1:
            image = image_list[0]
        else:
            # as lamm don't support multiimage input, we concat in horizon
            image = self.horizontal_concat(image_list) 
        return image

    def do_generate(self, image_list: list, prompt: str, max_new_tokens, **kwargs):
        outputs = self.model.generate({
            'prompt': [prompt],
            'images': [image_list],
            'top_p': 0.9,
            'temperature': 1.0,
            'max_tgt_len': max_new_tokens,
            'modality_embeds': []
        })
        return outputs[0].split('\n###')[0]

    def do_ppl(self, batch_images, conversations, batch_options, **kwargs):
        option_ids = []
        for option in batch_options:
            option_token = self.model.llama_tokenizer.encode(option, add_special_tokens=False, return_tensors="pt").squeeze(0)
            option_ids.append(option_token)

        logits, labels = self.model.ppl_forward(dict(
            vision_type = 'image',
            task_type = self.task_type,
            vision_paths = batch_images,
            output_texts = conversations,
        ))
        logits = logits[:,:-1]
        labels = labels[:,1:]
        results = []
        for idx in range(labels.shape[0]):
            option_len = len(option_ids[idx])
            non_zero_indices = torch.nonzero(labels[idx]!=-100, as_tuple=False).squeeze()
            end_index = non_zero_indices.max() - 2
            start_index = end_index - option_len
            if not np.all(labels[idx][start_index: end_index].detach().cpu().numpy() == option_ids[idx][-option_len:].numpy()):
                import ipdb;ipdb.set_trace()
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, option_ids[idx][-option_len:]]).mean().item()
            results.append(score)
        return results

    @torch.no_grad()
    def do_generate_3d(self, modality_inputs, question_list, max_new_tokens=128):
        modality_inputs.update({
            'top_p': 0.9,
            'temperature': 1.0,
            'max_tgt_len': max_new_tokens,
            'modality_embeds': [],
            'prompt': question_list,
        })
        outputs = self.model.generate(modality_inputs)
        return [output.split('\n###')[0] for output in outputs]

    @torch.no_grad()
    def batch_generate_3d(self, modality_inputs, question_list, max_new_tokens=128, sys_msg = None, **kwargs):
        prompts = self.generate_conversation_text(question_list, sys_msg = sys_msg)
        outputs = self.do_generate_3d(modality_inputs, prompts, max_new_tokens)
        return outputs
