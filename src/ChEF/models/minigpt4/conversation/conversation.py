import argparse
import time
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from ..common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = stops
        self.prompt_len = 0

    def _contains_subsequence(self, large_tensor, small_tensor):
        len_small = len(small_tensor)
        for i in range(0, len(large_tensor)-len_small+1):
            flag = torch.all((small_tensor == large_tensor[i: i+len_small])).item()
            if flag:
                return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for x in input_ids:
            end_now = False
            for stop in self.stops:
                stop = stop.to(x.device)
                end_now |= self._contains_subsequence(x[self.prompt_len:], stop)
                # if torch.all((stop == input_ids[i][-len(stop):])).item():
                #     return True
            if not end_now:
                return False
        return True


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=self.stop_words_ids)])
    
    def move_stopping_criteria_device(self, device, dtype=torch.float32):
        self.stop_words_ids = [stop_tensor.to(device, dtype=dtype) for stop_tensor in self.stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=self.stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=False,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        # print(f'Check the shape of image emb: {image_emb.shape}')
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def get_context_emb(self, conv, img_list, answer = None):
        prompt = conv.get_prompt()
        if answer is not None:
            prompt += answer
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def batch_answer(self, image_list, question_list, chat_list, max_new_tokens=300, num_beams=5, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=-1.0, temperature=1.0, max_length=2000):
        embs_list = []
        for image, question, conv in zip(image_list, question_list, chat_list):
            img_list = []
            self.upload_img(image, conv, img_list)
            self.ask(question, conv)
            conv.append_message(conv.roles[1], None)
            embs = self.get_context_emb(conv, img_list)
            embs_list.append(embs)
        max_emb_token = max([x.shape[1] for x in embs_list])
        embs_list = torch.cat([F.pad(x, (0, 0, max_emb_token - x.shape[1], 0, 0, 0), value=0) for x in embs_list], dim=0)

        assert max_emb_token + max_new_tokens < max_length
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs_list,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=False,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        #import ipdb;ipdb.set_trace()
        batch_outputs = []
        for output_token in outputs:
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            batch_outputs.append(output_text)
        return batch_outputs

    def ppl_answer(self, image_list, question_list, chat_list, answer_list, answer_options, CoT_list = None, calib = False):
        embs_list = []
        for idx,(image, question, conv, answer) in enumerate(zip(image_list, question_list, chat_list, answer_list)):
            img_list = []
            self.upload_img(image, conv, img_list)
            self.ask(question, conv)
            conv.append_message(conv.roles[1], None)
            if CoT_list is not None:
                embs = self.get_context_emb(conv, img_list, answer = CoT_list[idx] + answer)
            else:
                embs = self.get_context_emb(conv, img_list, answer = answer)
            embs_list.append(embs)
        results = self.do_ppl(embs_list, answer_list, answer_options, calib = calib)
        
        return results
      
    def icl_batch_answer(self, image_list, question_list, chat_list, ice_imgs_emb, sample_data, incontext_cfg, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        embs_list = []
        prompts = []
        for i, (image, question, conv) in enumerate(zip(image_list, question_list, chat_list)):
            img_list = []
            self.upload_img(image, conv, img_list)
            img_list = self.get_icl_prompt_img(question, conv, img_list, sample_data[i], ice_imgs_emb, i, incontext_cfg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            embs = self.get_context_emb(conv, img_list)
            embs_list.append(embs)
        max_emb_token = max([x.shape[1] for x in embs_list])
        embs_list = torch.cat([F.pad(x, (0, 0, max_emb_token - x.shape[1], 0, 0, 0), value=0) for x in embs_list], dim=0)

        assert max_emb_token + max_new_tokens < max_length
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs_list,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        batch_outputs = []
        for output_token in outputs:
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            batch_outputs.append(output_text)
        return batch_outputs, prompts
    
    def get_imgs_emb(self, ice_imgs):
        img_list = []
        if type(ice_imgs) != list:
            ice_imgs = [ice_imgs]
        for image in ice_imgs:
            if isinstance(image, str):  # is a image path
                raw_image = Image.open(image).convert('RGB')
                image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            elif isinstance(image, Image.Image):
                raw_image = image
                image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image = image.to(self.device)

            image_emb, _ = self.model.encode_img(image)
            # print(f'Check the shape of image emb: {image_emb.shape}')
            img_list.append(image_emb)
        return img_list
    
    def get_icl_prompt_img(self, question, conv, img_list, ice, ice_imgs_emb, index, incontext_cfg):
        if incontext_cfg['add_sysmsg']:
            conv.system += incontext_cfg['sysmsg']
        if incontext_cfg['use_pic']:
            img_list = ice_imgs_emb[index] + img_list
            if incontext_cfg['mult_conversations']:
                for i in range(len(ice_imgs_emb[index])):
                    if not isinstance(ice[i]['gt_answers'], list):
                        conv.messages[-1][-1] += ice[i]['question']
                        conv.append_message(conv.roles[1], ice[i]['gt_answers'])
                    else:
                        conv.messages[-1][-1] += ice[i]['question']
                        conv.append_message(conv.roles[1], ice[i]['gt_answers'][0])
                    conv.append_message(conv.roles[0], '<Img><ImageHere></Img>')
                conv.messages[-1][-1] += question
            else:
                for i in range(len(ice_imgs_emb[index])):
                    if not isinstance(ice[i]['gt_answers'], list):
                        conv.messages[-1][-1] += f"{ice[i]['question']}: {ice[i]['gt_answers']}."
                    else:
                        conv.messages[-1][-1] += f"{ice[i]['question']}: {ice[i]['gt_answers'][0]}."
                    conv.messages[-1][-1] += '<Img><ImageHere></Img>'
                conv.messages[-1][-1] += question
        else:
            icl_question = ''
            for j in range(incontext_cfg['ice_num']):
                if not isinstance(ice[j]['gt_answers'], list):
                    icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers']}. "
                else:
                    icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers'][0]}. "
            icl_question += f"{question}: "
            self.ask(icl_question, conv)
    
    def icl_ppl_batch_answer(self, image_list, question_list, chat_list, answer_list, answer_options, ice_imgs_emb, sample_data, incontext_cfg, CoT_list = None):
        embs_list = []
        prompts = []
        for i, (image, question, conv, answer) in enumerate(zip(image_list, question_list, chat_list, answer_list)):
            img_list = []
            self.upload_img(image, conv, img_list)
            img_list = self.get_icl_prompt_img(question, conv, img_list, sample_data[i], ice_imgs_emb, i, incontext_cfg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            if CoT_list is not None:
                embs = self.get_context_emb(conv, img_list, answer = CoT_list[i] + answer)
            else:
                embs = self.get_context_emb(conv, img_list, answer = answer)
            embs_list.append(embs)

        results = self.do_ppl(embs_list, answer_list, answer_options)

        return results, prompts
    
    def do_ppl(self, embs_list, answer_list, answer_options, calib = False):
        max_emb_token = max([x.shape[1] for x in embs_list])
        padding_len_list = [max_emb_token - x.shape[1] for x in embs_list]
        target_ids = []
        answer_start_indices = []
        answer_end_indices = []
        answer_token_list = []
        template_token_list = []
        for template, option in zip(answer_list, answer_options):
            template_token = self.model.llama_tokenizer(template, return_tensors='pt', add_special_tokens=False).input_ids
            template_token_list.append(template_token)
            option_token = self.model.llama_tokenizer(option, return_tensors='pt', add_special_tokens=False).input_ids
            target_ids.append(template_token)
            token_len = len(option_token[0])
            for index in range(len(template_token[0])):
                if torch.all(template_token[0][index: index + token_len] == option_token[0]):
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token[0])
                    break
            assert len(answer_start_indices) == len(template_token_list), "tokenizer encode answer in template different from answer only"

        target_ids = torch.cat([F.pad(x, (max_emb_token - x.shape[1], 0,0,0), value = -100) for x in target_ids], dim=0).to(self.device)
        embs_list = torch.cat([F.pad(x, (0, 0, max_emb_token - x.shape[1], 0, 0, 0), value=0) for x in embs_list], dim=0) # left padding
        att_mask = torch.ones(embs_list.shape[:-1])
        for idx, padding_len in enumerate(padding_len_list):
            att_mask[idx,:padding_len] = 0
        att_mask = att_mask.bool().to(self.device)
        
        outputs = self.model.llama_model(
            inputs_embeds = embs_list,
            attention_mask = att_mask,
            return_dict = True
        )
        logits = outputs['logits'][:,:-1]
        target_ids = target_ids[:,1:]
        
        loss_mask = target_ids!=-100
        results = []
        if calib:
            for idx, item_logits in enumerate(logits):
                score = 0.0
                item_prob = F.softmax(item_logits[loss_mask[idx]][answer_start_indices[idx]: answer_end_indices[idx]], dim=-1)
                for jdx in range(answer_end_indices[idx]-answer_start_indices[idx]):
                    score += torch.log(item_prob[jdx, answer_token_list[idx][jdx]]).item()
                score = score/len(answer_token_list[idx])
                results.append(score)
        else:
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), target_ids.reshape(-1),ignore_index=-100, reduction='none')
            loss = loss.reshape(-1, target_ids.shape[1]).float()
            for idx, item_loss in enumerate(loss):
                results.append(item_loss[loss_mask[idx]][answer_start_indices[idx]: answer_end_indices[idx]].mean().item())
        return results
