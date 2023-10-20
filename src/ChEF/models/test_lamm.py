import torch
import yaml
from .lamm.openlamm import LAMMPEFTModel, process_batch_instance
from .lamm.utils.conversations import conv_templates
from .utils import get_image
import torch.nn.functional as F
from .test_base import TestBase


class TestLAMM(TestBase):
    def __init__(self, 
                 model_path,
                 vicuna_ckpt_path,
                 cfg_path = 'models/lamm/lamm_eval.yaml',
                 encoder_ckpt_path = None, 
                 task_type = 'normal',
                 **kwargs
                 ):
        self.conv_mode = 'simple'
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg_dict['vicuna_ckpt_path'] = vicuna_ckpt_path
        if encoder_ckpt_path is not None:
            cfg_dict['encoder_ckpt_path'] = encoder_ckpt_path
        self.model = LAMMPEFTModel(**cfg_dict)
        delta_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)
        self.model = self.model.eval().half()
        self.task_type = task_type
        self.move_to_device()

    def move_to_device(self):
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)

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
    
    def generate_icl_text(self, input_list, ices, incontext_cfg):
        conv = conv_templates[self.conv_mode]
        prompts_list = []
        for input, ice in zip(input_list, ices):
            prompts = ''
            prompts += conv.system
            if incontext_cfg['add_sysmsg']:
                prompts += incontext_cfg['sysmsg']
            if incontext_cfg['use_pic']:
                raise NotImplementedError
            else:
                icl_question = ''
                for j in range(incontext_cfg['ice_num']):
                    if not isinstance(ice[j]['gt_answers'], list):
                        icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers']}. "
                    else:
                        icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers'][0]}. "
                icl_question += f"{input}: "
            prompts += "{} {}: {}\n".format(conv.sep, conv.roles[0], icl_question)
            prompts_list.append(prompts)
        return prompts_list

    @torch.no_grad()
    def do_generate(self, images, question_list, max_new_tokens=128):
        outputs = self.model.generate({
            'prompt': question_list,
            'images': images,
            'top_p': 0.9,
            'temperature': 1.0,
            'max_tgt_len': max_new_tokens,
            'modality_embeds': []
        })
        return [output.split('\n###')[0] for output in outputs]

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        image = [get_image(image)]
        text = self.generate_conversation_text([question])
        outputs = self.do_generate(image, text, max_new_tokens)
        return outputs[0]
        
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        images = [get_image(image) for image in image_list]
        prompts = self.generate_conversation_text(question_list)
        outputs = self.do_generate(images, prompts, max_new_tokens)
        return outputs
    
    @torch.no_grad()
    def icl_batch_generate(self, image_list, question_list, ices, incontext_cfg, max_new_tokens=128):
        images = [get_image(image) for image in image_list]
        prompts = self.generate_icl_text(question_list, ices, incontext_cfg)
        outputs = self.do_generate(images, prompts, max_new_tokens)
        return outputs, prompts


    @torch.no_grad()
    def ppl_inference(self, image_list, question_list, answer_list, answer_pool, CoT_list = None, calib = False):
        images = []
        images = [get_image(image) for image in image_list]
        conversations = []
        for idx, (question, answer) in enumerate(zip(question_list, answer_list)):
            conversation = []
            conversation.append({'from':'human', 'value': question})
            fromgpt = answer
            if CoT_list is not None:
                fromgpt = CoT_list[idx] + '\n' + fromgpt
                answer_list[idx] = fromgpt
            conversation.append({'from':'gpt', 'value': fromgpt})
            conversations.append(conversation)
    
        results = self.do_ppl(images, conversations, answer_list, answer_pool, calib = calib)
        return results

    @torch.no_grad()
    def icl_ppl_inference(self, image_list, question_list, answer_list, answer_pool, ices, incontext_cfg, CoT_list = None, calib = False):
        images = []
        images = [get_image(image) for image in image_list]
        if incontext_cfg['use_pic']:
            raise NotImplementedError
        conversations = []

        for idx, (question, answer, ice) in enumerate(zip(question_list, answer_list, ices)):
            if incontext_cfg['add_sysmsg']:
                icl_sysmsg = incontext_cfg['sysmsg']
            else:
                icl_sysmsg = None
            conversation = []
            icl_question = ''
            for j in range(incontext_cfg['ice_num']):
                if not isinstance(ice[j]['gt_answers'], list):
                    icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers']}. "
                else:
                    icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers'][0]}. "
            icl_question += f"{question}: "
            conversation.append({'from':'human', 'value': icl_question})
            fromgpt = answer
            if CoT_list is not None:
                fromgpt = CoT_list[idx] + '\n' + fromgpt
                answer_list[idx] = fromgpt
            conversation.append({'from':'gpt', 'value': fromgpt})
            conversations.append(conversation)

        results = self.do_ppl(images, conversations, answer_list, answer_pool, calib = calib, icl_sysmsg=icl_sysmsg)
        return results, conversations
    
    @torch.no_grad()
    def do_ppl(self, images, conversations, answer_list, answer_pool, calib = False, icl_sysmsg=None):
        answer_start_indices = []
        answer_end_indices = []
        answer_token_list = []
        template_token_list = []
        for template, option in zip(answer_list, answer_pool):
            template_token = self.model.llama_tokenizer.encode(template, add_special_tokens=False)
            template_token_list.append(template_token)
            option_token = self.model.llama_tokenizer.encode(option, add_special_tokens=False)
            token_len = len(option_token)
            for index in range(len(template_token)):
                if template_token[index: index + token_len] == option_token:
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token)
                    break
            assert len(answer_start_indices) == len(template_token_list), "tokenizer encode answer in template different from answer only"
            
        logits, target_ids = self.model(dict(
            vision_type = 'image',
            task_type = self.task_type,
            vision_paths = images,
            output_texts = conversations,
            icl_sysmsg = icl_sysmsg
        ))
        logits = logits[:,:-1]
        target_ids = target_ids[:,1:]
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
    def do_calibration(self, image_list, question_list, answer_list, answer_pool, CoT_list=None):
        return super().do_calibration(image_list, question_list, answer_list, answer_pool, CoT_list)
