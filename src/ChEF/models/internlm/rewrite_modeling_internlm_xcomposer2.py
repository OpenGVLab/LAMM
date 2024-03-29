from typing import List, Tuple
import torch
from .modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM


class RewriteInternLMXComposer2ForCausalLM(InternLMXComposer2ForCausalLM):
    meta_instruction = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
    '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
    '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
    '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.'

    def img2emb(self, image_list):
        image_tensor, atts_img_list, img_target_list = [], [], []
        for image in image_list:
            img_embeds = self.vision_proj(self.vit(image.unsqueeze(0)))
            image_tensor.append(img_embeds)
            atts_img = torch.ones(
                img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)
            atts_img_list.append(atts_img)
            img_target = torch.ones(
                img_embeds.size()[:2], dtype=torch.long).to(
                    img_embeds.device) * -100
            img_target_list.append(img_target)
        image_tensor = torch.cat(image_tensor, dim=1) # bs, seq_len, dim
        atts_img_list = torch.cat(atts_img_list, dim=1) # bs, seq_len, dim
        img_target_list = torch.cat(img_target_list, dim=1) # bs, seq_len, dim
        return image_tensor, atts_img_list, img_target_list
    
    def build_inputs(self, query: str, history: List[Tuple[str, str]] = ...):
        prompt = f"""[UNUSED_TOKEN_146]system\n{self.meta_instruction}[UNUSED_TOKEN_145]\n"""
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
        return prompt
    
    def interleav_wrap_chat(self, tokenizer, prompt, image):
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
        wrap_token_ids = []
        for image, text in zip(img_list, text_list):
            img_embeds, atts_img, img_target = self.img2emb(image)
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

            wrap_tokens = torch.cat(wrap_tokens, dim=1).to(self.device)
            wrap_embeds = torch.cat(wrap_embeds, dim=1).to(self.device)
            wrap_atts = torch.cat(wrap_atts, dim=1).to(self.device)
            wrap_im_mask = torch.cat(wrap_im_mask, dim=1).to(self.device)
            wrap_token_ids.append(wrap_tokens)

            wrap_target = self.mask_human_targets(wrap_tokens).to(self.device)

            wrap_embeds_list.append(wrap_embeds)
            wrap_atts_list.append(wrap_atts)
            wrap_target_list.append(wrap_target)
            wrap_im_mask_list.append(wrap_im_mask)

        wrap_embeds_list = [item.squeeze(0) for item in wrap_embeds_list]
        wrap_embeds_list = torch.nn.utils.rnn.pad_sequence(
            wrap_embeds_list,
            batch_first=True,
            padding_value=0.) # right padding
        
        wrap_atts_list = [item.squeeze(0) for item in wrap_atts_list]
        wrap_atts_list = torch.nn.utils.rnn.pad_sequence(
            wrap_atts_list,
            batch_first=True,
            padding_value=0.)
        
        wrap_target_list = [item.squeeze(0) for item in wrap_target_list]
        wrap_target_list = torch.nn.utils.rnn.pad_sequence(
            wrap_target_list,
            batch_first=True,
            padding_value=-100)
        
        wrap_im_mask_list = [item.squeeze(0) for item in wrap_im_mask_list]
        wrap_im_mask_list = torch.nn.utils.rnn.pad_sequence(
            wrap_im_mask_list,
            batch_first=True,
            padding_value=0.)
        
        wrap_token_ids = [item.squeeze(0) for item in wrap_token_ids]
        wrap_token_ids = torch.nn.utils.rnn.pad_sequence(
            wrap_token_ids,
            batch_first=True,
            padding_value=0.)
        return wrap_embeds_list, wrap_atts_list, wrap_target_list, wrap_im_mask_list, wrap_token_ids