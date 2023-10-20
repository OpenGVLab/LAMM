import torch
from transformers import CLIPImageProcessor
from .otter.modeling_otter import OtterForConditionalGeneration
from .instruct_blip.models.eva_vit import convert_weights_to_fp16
from .utils import get_image
from .utils import Conversation, SeparatorStyle
from .test_base import TestBase
import torch.nn.functional as F

CONV_VISION = Conversation(
    system='',
    roles=("<image>User", "GPT"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep=" ",
)

class TestOtter(TestBase):
    def __init__(self, model_path, **kwargs) -> None:
        self.model = OtterForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()
        self.tokenizer.padding_side = "left"
        self.ice_imgs_emb = None
        self.move_to_device()

    def move_to_device(self):
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
            convert_weights_to_fp16(self.model.vision_encoder)
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model.vision_encoder = self.model.vision_encoder.to(self.device, dtype=self.dtype)
        self.model = self.model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        image = get_image(image)
        vision_x = (self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
        lang_x = self.model.text_tokenizer([f"<image> User: {question} GPT: <answer>"], return_tensors="pt")
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device, dtype=self.dtype),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        output = self.model.text_tokenizer.decode(generated_text[0])
        output = [x for x in output.split(' ') if not x.startswith('<')]
        out_label = output.index('GPT:')
        output = ' '.join(output[out_label + 1:])
        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.image_processor.preprocess([x], return_tensors="pt")["pixel_values"].unsqueeze(0) for x in imgs]
        vision_x = (torch.stack(imgs, dim=0))
        prompts = [f"<image> User: {question} GPT: <answer>" for question in question_list]
        lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device, dtype=self.dtype),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        total_output = []
        for i in range(len(generated_text)):
            output = self.model.text_tokenizer.decode(generated_text[i])
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = output.index('GPT:')
            output = ' '.join(output[out_label + 1:])
            total_output.append(output)
        return total_output

    @torch.no_grad()
    def do_ppl(self, vision_x, lang_x, answer_list, answer_pool, calib = False):
        answer_start_indices = []
        answer_end_indices = []
        template_token_list = []
        answer_token_list = []
        for template, option in zip(answer_list, answer_pool):
            template_token = self.tokenizer.encode(template)[1:] # skip <s>
            option_token = self.tokenizer.encode(option)[1:]
            token_len = len(option_token)
            template_token_list.append(template_token)
            for index in range(len(template_token)):
                if template_token[index: index + token_len] == option_token:
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token)
                    break
            assert len(answer_start_indices) == len(template_token_list), "tokenizer encode answer in template different from answer only"
        
        input_ids=lang_x["input_ids"].to(self.model.device)
        attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype)
        output = self.model(vision_x=vision_x, lang_x=input_ids, attention_mask=attention_mask)
        logits = output['logits'][:,:-1]
        input_ids = input_ids[:,1:]

        start_indices, end_indices = [], []
        for i in range(len(answer_list)):
            token_len = len(template_token_list[i])
            for index in range(input_ids.shape[1]-token_len, 0, -1):
                if input_ids[i,index: index+token_len].cpu().numpy().tolist() == template_token_list[i]:
                    start_indices.append(index + answer_start_indices[i])
                    end_indices.append(index + answer_end_indices[i])
                    input_ids[i,:index] = -100
                    break
            assert len(start_indices) == (i+1), "tokenizer encode answer different from answer in conversation"
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
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), input_ids.reshape(-1),ignore_index=-100, reduction='none')
            loss = loss.reshape(-1, input_ids.shape[1]).float()
            for idx, item_loss in enumerate(loss):
                results.append(item_loss[start_indices[idx]:end_indices[idx]].mean().item())
        return results

    @torch.no_grad()
    def ppl_inference(self, image_list, question_list, answer_list, answer_pool, CoT_list = None, calib = False):
        imgs = [get_image(img) for img in image_list]
        imgs = [self.image_processor.preprocess([x], return_tensors="pt")["pixel_values"].unsqueeze(0) for x in imgs]
        vision_x = (torch.stack(imgs, dim=0))
        vision_x=vision_x.to(self.model.device, dtype=self.dtype)
        prompts = [f"<image> User: {question} GPT: <answer>" for question in question_list]
        if CoT_list is not None:
            prompts = [prompt + ' ' + cot + '\n' for prompt, cot in zip(prompts, CoT_list)]
        prompts = [prompt + ' ' + answer for prompt, answer in zip(prompts, answer_list)]
        lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
        results = self.do_ppl(vision_x, lang_x, answer_list, answer_pool, calib)
        return results
    
    def get_icl_prompt(self, question_list, ices, chat_list, incontext_cfg):
        prompts =[]
        for question, conv, ice in zip(question_list, chat_list, ices):
            if incontext_cfg['add_sysmsg']:
                conv.system += incontext_cfg['sysmsg']
            conv.append_message(conv.roles[0], '')
            if incontext_cfg['use_pic']:
                if incontext_cfg['mult_conversations']:
                    for i in range(incontext_cfg['ice_num']):
                        if not isinstance(ice[i]['gt_answers'], list):
                            conv.messages[-1][-1] += ice[i]['question']
                            conv.append_message(conv.roles[1], '<answer>' + ice[i]['gt_answers'] + '<|endofchunk|>')
                        else:
                            conv.messages[-1][-1] += ice[i]['question']
                            conv.append_message(conv.roles[1], '<answer>' + ice[i]['gt_answers'][0] + '<|endofchunk|>')
                        conv.append_message(conv.roles[0], '')
                    conv.messages[-1][-1] += question
                else:
                    for i in range(incontext_cfg['ice_num']):
                        if not isinstance(ice[i]['gt_answers'], list):
                            conv.messages[-1][-1] += f"{ice[i]['question']}: {ice[i]['gt_answers']}. "
                        else:
                            conv.messages[-1][-1] += f"{ice[i]['question']}: {ice[i]['gt_answers'][0]}. "
                    conv.messages[-1][-1] += question
            else:
                icl_question = ''
                for j in range(incontext_cfg['ice_num']):
                    if not isinstance(ice[j]['gt_answers'], list):
                        icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers']}. "
                    else:
                        icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers'][0]}. "
                icl_question += f"{question}: "
                conv.messages[-1][-1] += icl_question
            
            conv.append_message(conv.roles[1], '<answer>')
            prompt = conv.get_prompt()
            prompts.append(prompt)

        return prompts
    
    @torch.no_grad()
    def get_ice_imgs(self, sample_data):
        imgs_with_ice = []
        for ice_data in sample_data:
            ice_images = []
            for ice in ice_data:
                ice_image = ice['image_path']
                ice_image = get_image(ice_image)
                ice_images.append(ice_image)
            ice_images = [self.image_processor(image, return_tensors='pt')["pixel_values"].unsqueeze(0) for image in ice_images]
            imgs_with_ice.append(ice_images)
        self.ice_imgs_emb = imgs_with_ice

    @torch.no_grad()
    def icl_batch_generate(self, image_list, question_list, ices, incontext_cfg, max_new_tokens=128):
        sample_data = ices
        imgs = [get_image(img) for img in image_list]
        imgs = [self.image_processor.preprocess([x], return_tensors="pt")["pixel_values"].unsqueeze(0) for x in imgs]
        
        if incontext_cfg['use_pic']:
            self.get_ice_imgs(sample_data, incontext_cfg)
            imgs_with_ice = []
            for i, img in enumerate(imgs):
                img_with_ice = []
                img_with_ice.extend(self.ice_imgs_emb[i])
                img_with_ice.append(img)
                imgs_with_ice.append(torch.cat(img_with_ice, dim=0))
            imgs = imgs_with_ice

        chat_list = [CONV_VISION.copy() for _ in range(len(question_list))]
        prompts = self.get_icl_prompt(question_list, sample_data, chat_list, incontext_cfg)
        vision_x = (torch.stack(imgs, dim=0))
        lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device, dtype=self.dtype),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        total_output = []
        for i in range(len(generated_text)):
            output = self.model.text_tokenizer.decode(generated_text[i])
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = len(output) - output[::-1].index('GPT:') - 1
            output = ' '.join(output[out_label + 1:])
            total_output.append(output)
        return total_output, prompts
    
    @torch.no_grad()
    def icl_ppl_inference(self, image_list, question_list, answer_list, answer_pool, ices, incontext_cfg, CoT_list = None):
        sample_data = ices
        imgs = [get_image(img) for img in image_list]
        imgs = [self.image_processor.preprocess([x], return_tensors="pt")["pixel_values"].unsqueeze(0) for x in imgs]
        
        if incontext_cfg['use_pic']:
            self.get_ice_imgs(sample_data)
            imgs_with_ice = []
            for i, img in enumerate(imgs):
                img_with_ice = []
                img_with_ice.extend(self.ice_imgs_emb[i])
                img_with_ice.append(img)
                imgs_with_ice.append(torch.cat(img_with_ice, dim=0))
            imgs = imgs_with_ice

        chat_list = [CONV_VISION.copy() for _ in range(len(question_list))]
        prompts = self.get_icl_prompt(question_list, sample_data, chat_list, incontext_cfg)

        vision_x = (torch.stack(imgs, dim=0))
        vision_x=vision_x.to(self.model.device, dtype=self.dtype)

        if CoT_list is not None:
            prompts = [prompt + ' ' + cot + '\n' for prompt, cot in zip(prompts, CoT_list)]
        prompts = [prompt + ' ' + answer for prompt, answer in zip(prompts, answer_list)]
        lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
        results = self.do_ppl(vision_x, lang_x, answer_list, answer_pool)
        return results, prompts


    def do_calibration(self, image_list, question_list, answer_list, answer_pool, CoT_list=None):
        return super().do_calibration(image_list, question_list, answer_list, answer_pool, CoT_list)