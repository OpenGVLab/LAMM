import torch

from .mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from .mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from .utils import get_image, Conversation, SeparatorStyle
import torch.nn.functional as F
from .test_base import TestBase

prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI:"

CONV_VISION = Conversation(
    system="The following is a conversation between a curious human and AI assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("Human", "AI"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
)



class TestMplugOwl(TestBase):
    def __init__(self, model_path, **kwargs):
        self.model = MplugOwlForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.ice_imgs_emb = None
        self.model.eval()
        self.move_to_device()
        
    def move_to_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.device = 'cpu'
            self.dtype = torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        prompts = [prompt_template.format(question)]
        image = get_image(image)
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': max_new_tokens
        }

        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        images = [get_image(image) for image in image_list]
        images = [self.image_processor(image, return_tensors='pt').pixel_values for image in images]
        images = torch.cat(images, dim=0).to(self.device, dtype=self.dtype)
        prompts = [prompt_template.format(question) for question in question_list]
        inputs = self.processor(text=prompts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = images
        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': max_new_tokens
        }

        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in res.tolist()]
        return outputs

    @torch.no_grad()
    def ppl_inference(self, image_list, question_list, answer_list, answer_options, CoT_list = None, calib = False):
        images = [get_image(image) for image in image_list]
        images = [self.image_processor(image, return_tensors='pt').pixel_values for image in images]
        images = torch.cat(images, dim=0).to(self.device, dtype=self.dtype)
        prompts = [prompt_template.format(question) for question in question_list]
        results = self.do_ppl(images, prompts, answer_list, answer_options, CoT_list = CoT_list, calib = calib)
        return results


    @torch.no_grad()
    def icl_batch_generate(self, image_list, question_list, ices, incontext_cfg, max_new_tokens=128):
        sample_data = ices
        if incontext_cfg['retriever_type'] != 'fixed' and incontext_cfg['use_pic']:
            imgs_with_ice = []
            for i, img in enumerate(image_list):
                image_path = []
                for ice in ices[i]:
                    image_path.append(ice['image_path'])
                imgs_with_ice.extend(image_path)
                imgs_with_ice.append(img)
            image_list = imgs_with_ice
        elif incontext_cfg['use_pic'] and self.ice_imgs_emb is None:
            self.get_ice_imgs_emb(ices, incontext_cfg)

        images = [get_image(image) for image in image_list]
        images = [self.image_processor(image, return_tensors='pt').pixel_values for image in images]
        images = torch.cat(images, dim=0).to(self.device, dtype=self.dtype)
        
        chat_list = [CONV_VISION.copy() for _ in range(len(question_list))]
        prompts = self.get_icl_prompt(question_list, sample_data, chat_list, incontext_cfg)
        inputs = self.processor(text=prompts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = images

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': max_new_tokens
        }

        with torch.no_grad():
            res = self.model.icl_generate(self.ice_imgs_emb, **inputs, **generate_kwargs)
        outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in res.tolist()]

        return outputs, prompts
    
    @torch.no_grad()
    def get_ice_imgs_emb(self, ices, incontext_cfg):
        ice_images = [ice['image_path'] for ice in ices[0]]
        ice_images = [get_image(image) for image in ice_images]
        ice_images = [self.image_processor(image, return_tensors='pt').pixel_values for image in ice_images]
        ice_images = torch.cat(ice_images, dim=0).to(self.device, dtype=self.dtype)
        ice_images = ice_images.to(self.model.vision_model.embeddings.cls_token.data.dtype)

        # self.ice_imgs_emb = self.chat.get_ice_imgs_emb(ice_images)
        with torch.no_grad():
            image_embeds = self.model.vision_model(ice_images, return_dict=True).last_hidden_state
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
            )
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.model.abstractor(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            query_output = query_outputs["last_hidden_state"]
            self.ice_imgs_emb = query_output


    def get_icl_prompt(self, question_list, ices, chat_list, incontext_cfg):
        prompts =[]
        for question, conv, ice in zip(question_list, chat_list, ices):
            if incontext_cfg['add_sysmsg']:
                conv.system += incontext_cfg['sysmsg']
            conv.append_message(conv.roles[0], "<image>")
            conv.append_message(conv.roles[0], '')
            if incontext_cfg['use_pic']:
                if incontext_cfg['mult_conversations']:
                    for i in range(incontext_cfg['ice_num']):
                        if not isinstance(ice[i]['gt_answers'], list):
                            conv.messages[-1][-1] += ice[i]['question']
                            conv.append_message(conv.roles[1], ice[i]['gt_answers'])
                        else:
                            conv.messages[-1][-1] += ice[i]['question']
                            conv.append_message(conv.roles[1], ice[i]['gt_answers'][0])
                        conv.append_message(conv.roles[0], '<image>')
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
            
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)

        return prompts
        
    @torch.no_grad()
    def icl_ppl_inference(self, image_list, question_list, answer_list, answer_options, ices, incontext_cfg, CoT_list=None, calib = False):
        sample_data = ices
        if incontext_cfg['retriever_type'] != 'fixed' and incontext_cfg['use_pic']:
            imgs_with_ice = []
            for i, img in enumerate(image_list):
                image_path = []
                for ice in ices[i]:
                    image_path.append(ice['image_path'])
                imgs_with_ice.extend(image_path)
                imgs_with_ice.append(img)
            image_list = imgs_with_ice
        elif incontext_cfg['use_pic'] and self.ice_imgs_emb is None:
            self.get_ice_imgs_emb(ices, incontext_cfg)

        images = [get_image(image) for image in image_list]
        images = [self.image_processor(image, return_tensors='pt').pixel_values for image in images]
        images = torch.cat(images, dim=0).to(self.device, dtype=self.dtype)
        chat_list = [CONV_VISION.copy() for _ in range(len(question_list))]
        prompts = self.get_icl_prompt(question_list, sample_data, chat_list, incontext_cfg)
        inferencer_type = 'icl_ppl'
        outputs = self.do_ppl(images, prompts, answer_list, answer_options, inferencer_type, CoT_list, calib = calib)

        return outputs, prompts
    
    @torch.no_grad()
    def do_calibration(self, image_list, question_list, answer_list, answer_pool, CoT_list=None):
        return super().do_calibration(image_list, question_list, answer_list, answer_pool, CoT_list)

    @torch.no_grad()
    def do_ppl(self, images, prompts, answer_list, answer_options, inferencer_type = 'ppl', CoT_list = None, calib = False):
        template_token_list = []
        answer_start_indices = []
        answer_end_indices = []
        answer_token_list = []
        for idx, (template, option) in enumerate(zip(answer_list, answer_options)):
            if CoT_list is not None:
                prompts[idx] += CoT_list[idx] + ' '
            prompts[idx] += template
            template_token = self.tokenizer.encode('\nAI:'+ template, return_tensors='pt', add_special_tokens=False)
            if CoT_list is not None:
                # if CoT, it's <COT> <prompt>
                template_token = self.tokenizer.encode(template, return_tensors='pt', add_special_tokens=False)
            option_token = self.tokenizer.encode(option, return_tensors='pt', add_special_tokens=False)
            template_token_list.append(template_token[0])
            token_len = len(option_token[0])
            # mplug decode ':A' different from 'A'
            if template == option:
                tmpidx = 0 if CoT_list is not None else 4
                answer_start_indices.append(tmpidx)
                answer_end_indices.append(tmpidx + token_len)
                answer_token_list.append(template_token[0][tmpidx: tmpidx + token_len])
                continue
            for index in range(len(template_token[0])):
                if torch.all(template_token[0][index: index + token_len] == option_token[0]):
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token[0])
                    break
            assert len(answer_start_indices) == len(template_token_list), "tokenizer encode answer in template different from answer only"
        
        inputs = self.processor(text=prompts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = images
        if inferencer_type == 'icl_ppl':
            outputs = self.model.icl_ppl_generate(num_images = torch.ones((len(images))) , ice_imgs_emb = self.ice_imgs_emb, **inputs)
        else:
            outputs = self.model.ppl_generate(num_images = torch.ones((len(images))) ,**inputs)
        logits = outputs['logits']
        target_ids = inputs['input_ids'].clone()

        logits = logits[:,:-1]
        target_ids = target_ids[:,1:]
        start_indices, end_indices = [], []
        for i in range(len(answer_list)):
            token_len = len(template_token_list[i]) - 1
            # shift one position. mplug decode ':\nAI:' different from '\nAI:'
            for index in range(target_ids.shape[1] - token_len, 1, -1):
                if torch.all(target_ids[i,index: index+token_len].cpu() == template_token_list[i][1:]):
                    start_indices.append(index + answer_start_indices[i] - 1)
                    end_indices.append(index + answer_end_indices[i] - 1)
                    target_ids[i,:index - 1] = -1
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
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), target_ids.reshape(-1),ignore_index=-1, reduction='none')
            loss = loss.reshape(-1, target_ids.shape[1]).float()
            for idx, item_loss in enumerate(loss):
                results.append(item_loss[start_indices[idx]: end_indices[idx]].mean().item())
        return results