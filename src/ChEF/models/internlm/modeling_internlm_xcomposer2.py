# # Copyright (c) InternLM. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch InternLMXComposer2 model."""
import copy
import queue
import threading
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from PIL import Image
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (add_start_docstrings_to_model_forward,
                                replace_return_docstrings)

try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None

from .build_mlp import build_vision_projector, build_vision_tower
from .configuration_internlm_xcomposer2 import InternLMXcomposer2Config
from .modeling_internlm2 import (InternLM2_INPUTS_DOCSTRING, InternLM2Model,
                                 InternLM2PreTrainedModel)

_CONFIG_FOR_DOC = 'InternLMXcomposer2Config'


class InternLMXComposer2ForCausalLM(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.tokenizer = None

        self.max_length = config.max_length
        print(f'Set max length to {self.max_length}')
        # Initialize weights and apply final processing
        self.post_init()

        self.vit = build_vision_tower()
        self.vision_proj = build_vision_projector()

        self.vis_processor = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InternLM2Model):
            module.gradient_checkpointing = value
        if value:
            self.vit.vision_tower.vision_model.encoder.gradient_checkpointing = value

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def encode_text(self, text, add_special_tokens=False):
        token = self.tokenizer(
            text, return_tensors='pt',
            add_special_tokens=add_special_tokens).input_ids.to(self.device)
        embs = self.model.tok_embeddings(token)
        return embs

    def encode_img(self, image):
        if image is None:
            return None
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = self.vis_processor(image).unsqueeze(0).to(self.device)
        else:
            assert isinstance(image, torch.Tensor)

        img_embeds, atts_img, img_target = self.img2emb(image)
        return img_embeds

    def img2emb(self, image):
        img_embeds = self.vision_proj(self.vit(image.to(self.device)))
        atts_img = torch.ones(
            img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        img_target = torch.ones(
            img_embeds.size()[:2], dtype=torch.long).to(
                img_embeds.device) * -100

        return img_embeds, atts_img, img_target

    def prompt_wrap(self, img_embeds, prompt):
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.tokenizer(
            p_before, return_tensors='pt',
            add_special_tokens=True).to(img_embeds.device)

        p_before_embeds = self.model.tok_embeddings(
            p_before_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds], dim=1)

        wrapped_atts_img = torch.ones(
            wrapped_img_embeds.size()[:-1],
            dtype=torch.long).to(img_embeds.device)

        wrapped_target = torch.ones(
            batch_size, wrapped_img_embeds.shape[1], dtype=torch.long).to(
                img_embeds.device) * -100

        return wrapped_img_embeds, wrapped_atts_img, wrapped_target

    def text2emb(self, text, add_special=False):
        to_regress_tokens = self.tokenizer(
            text,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            add_special_tokens=add_special).to(self.device)

        targets = self.mask_human_targets(to_regress_tokens.input_ids)
        targets = targets.to(self.device)
        return to_regress_tokens, targets

    def interleav_wrap_chat(self, tokenizer, query, image, history, meta_instruction):
        prompt = ''
        if meta_instruction:
            prompt += f"""[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""

        im_len = image.shape[1]
        image_nums = len(image)
        parts = prompt.split('<ImageHere>')
        wrap_embeds, wrap_im_mask = [], []
        temp_len = 0
    
        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = tokenizer(part, return_tensors='pt').to(self.device)
                part_embeds = self.model.tok_embeddings(
                    part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]))
                temp_len += part_embeds.shape[1]
            if idx < image_nums:
                wrap_embeds.append(image[idx].unsqueeze(0))
                wrap_im_mask.append(torch.ones(1, image[idx].shape[0]))
                temp_len += im_len
    
            if temp_len > self.max_length:
                break
    
        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)
        wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device)
        wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device).bool()
        inputs = {
            'inputs_embeds': wrap_embeds
        }
        return inputs, wrap_im_mask

    def interleav_wrap(self, img_list, text_list):
        wrap_embeds_list, wrap_atts_list = [], []
        wrap_target_list, wrap_im_mask_list = [], []

        for image, text in zip(img_list, text_list):
            img_embeds, atts_img, img_target = self.img2emb(image)
            text = text[0]
            parts = text.split('<ImageHere>')
            wrap_tokens, wrap_embeds, wrap_atts, wrap_im_mask = [], [], [], []
            temp_len = 0
            image_nums, im_len = img_embeds.shape[:2]
            need_bos = True
            for idx, part in enumerate(parts):
                if len(part) > 0:
                    part_tokens = self.tokenizer(
                        part,
                        return_tensors='pt',
                        padding='longest',
                        add_special_tokens=need_bos).to(self.device)
                    if need_bos:
                        need_bos = False
                    wrap_tokens.append(part_tokens.input_ids)
                    part_embeds = self.model.tok_embeddings(
                        part_tokens.input_ids)
                    wrap_embeds.append(part_embeds)
                    wrap_atts.append(part_tokens.attention_mask)
                    wrap_im_mask.append(
                        torch.zeros(part_embeds.shape[:2]).to(self.device))

                    temp_len += part_embeds.shape[1]
                if idx < image_nums:
                    wrap_tokens.append(img_target[idx].unsqueeze(0))
                    wrap_embeds.append(img_embeds[idx].unsqueeze(0))
                    wrap_atts.append(atts_img[idx].unsqueeze(0))
                    wrap_im_mask.append(
                        torch.ones_like(atts_img[idx].unsqueeze(0)))

                    temp_len += im_len
                if temp_len > self.max_length:
                    break

            wrap_tokens = torch.cat(wrap_tokens, dim=1)
            wrap_embeds = torch.cat(wrap_embeds, dim=1)
            wrap_atts = torch.cat(wrap_atts, dim=1)
            wrap_im_mask = torch.cat(wrap_im_mask, dim=1)

            wrap_target = self.mask_human_targets(wrap_tokens).to(self.device)

            wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device)
            wrap_atts = wrap_atts[:, :self.max_length].to(self.device)
            wrap_target = wrap_target[:, :self.max_length].to(self.device)
            wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device)

            wrap_embeds_list.append(wrap_embeds)
            wrap_atts_list.append(wrap_atts)
            wrap_target_list.append(wrap_target)
            wrap_im_mask_list.append(wrap_im_mask)

        wrap_embeds = torch.cat(wrap_embeds_list)
        wrap_atts = torch.cat(wrap_atts_list)
        wrap_target = torch.cat(wrap_target_list)
        wrap_im_mask = torch.cat(wrap_im_mask_list)
        return wrap_embeds, wrap_atts, wrap_target, wrap_im_mask

    def mask_human_targets(self, input_ids, pure=False):
        target_batch = []
        for bs in range(input_ids.shape[0]):
            ids = input_ids[bs]
            targets = copy.deepcopy(ids)
            end_count = 0
            last_eoa = 0
            for i, temp_id in enumerate(ids):
                if temp_id == 92542:
                    if end_count % 2 == 0:
                        targets[last_eoa:i + 6] = -100
                    else:
                        last_eoa = i + 1
                    end_count += 1
                # # eos and following pad
                elif temp_id == 2:
                    # loss on eos, but not on pad
                    targets[i + 1:] = -100
                    break
            # trunction, end at last question
            if temp_id != 2 and end_count % 2 == 0:
                # mask all after the last answer
                targets[last_eoa + 1:] = -100
            target_batch.append(targets.unsqueeze(0))
        target_batch = torch.cat(target_batch, dim=0)
        return target_batch

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """

        samples = kwargs.get('samples', None)
        if samples:
            if samples['data_type'][0] == 'text':
                has_img = False
            elif samples['data_type'][0] == 'multi':
                has_img = True
            else:
                raise NotImplementedError

            # encode text
            text = samples['text_input']
            # encode image
            if has_img:
                image = samples['image']
                to_regress_embeds, attention_mask, targets, im_mask = self.interleav_wrap(
                    image, text)
            else:
                to_regress_tokens, targets = self.text2emb(
                    text, add_special=True)
                to_regress_embeds = self.model.tok_embeddings(
                    to_regress_tokens.input_ids)
                attention_mask = to_regress_tokens.attention_mask
                im_mask = torch.zeros(to_regress_embeds.shape[:2]).cuda()

            inputs_embeds = to_regress_embeds[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            targets = targets[:, :self.max_length]
            im_mask = im_mask[:, :self.max_length].bool()
            labels = targets
        else:
            im_mask = kwargs.get('im_mask', None)
            if im_mask is None and inputs_embeds is not None:
                im_mask = torch.zeros(inputs_embeds.shape[:2]).to(
                    inputs_embeds.device)
                im_mask = im_mask.bool()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_mask=im_mask,
        )

        hidden_states = outputs[0]
        logits = self.output(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      im_mask=None,
                                      **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        im_mask = im_mask

        model_inputs.update({
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
            'im_mask': im_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past), )
        return reordered_past

    def build_inputs(self,
                     tokenizer,
                     query: str,
                     history: List[Tuple[str, str]] = [],
                     meta_instruction=''):
        prompt = ''
        if meta_instruction:
            prompt += f"""<s>[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
        else:
            prompt += '<s>'
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
        return tokenizer([prompt], return_tensors='pt')

    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        query: str,
        image: torch.Tensor = None,
        history: List[Tuple[str, str]] = [],
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float=1.005,
        meta_instruction:
        str = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
        '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.',
        **kwargs,
    ):
        if image is None:
            inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
            im_mask = torch.zeros(inputs['input_ids'].shape[:2]).cuda().bool()
        else:
            image = self.encode_img(image)
            inputs, im_mask = self.interleav_wrap_chat(tokenizer, query, image, history, meta_instruction)
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items() if torch.is_tensor(v)
        }
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
        ]
        outputs = self.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            im_mask=im_mask,
            **kwargs,
        )
        if image is None:
            outputs = outputs[0].cpu().tolist()[len(inputs['input_ids'][0]):]
        else:
            outputs = outputs[0].cpu().tolist()
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split('[UNUSED_TOKEN_145]')[0]
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = [],
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        **kwargs,
    ):
        """Return a generator in format: (response, history) Eg.

        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')]) ('你好，有什么可以帮助您的吗？', [('你好',
        '你好，有什么可以帮助您的吗？')])
        """
        if BaseStreamer is None:
            raise ModuleNotFoundError(
                'The version of `transformers` is too low. Please make sure '
                'that you have installed `transformers>=4.28.0`.')

        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(BaseStreamer):

            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ''
                self.received_inputs = False
                self.queue.put(
                    (self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError('ChatStreamer only supports batch size 1')
                elif len(value.shape) > 1:
                    value = value[0]

                if not self.received_inputs:
                    # The first received value is input_ids, ignore here
                    self.received_inputs = True
                    return

                token = self.tokenizer.decode([value[-1]],
                                              skip_special_tokens=True)
                if token.strip() != '[UNUSED_TOKEN_145]':
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()
