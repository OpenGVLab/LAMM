import torch
import torch.nn.functional as F
from .utils import get_image
from .shikra.builder.build_shikra import load_pretrained_shikra
from .shikra import model_args, training_args, quantization_kwargs
from .shikra.dataset.process_function import PlainBoxFormatter
from .shikra.dataset.builder import prepare_interactive
from .test_base import TestBase

class TestShikra(TestBase):
    def __init__(self, 
                 model_path,
                 encoder_ckpt_path = None, 
                 do_sample = False,
                 **kwargs
                 ):
        model_args.model_name_or_path = model_path
        if encoder_ckpt_path is not None:
            model_args.vision_tower = encoder_ckpt_path
        model, self.preprocessor = load_pretrained_shikra(model_args, training_args, **quantization_kwargs)
        if not getattr(model, 'is_quantized', False):
            model.to(dtype=torch.float16, device=torch.device('cuda'))
        if not getattr(model.model.vision_tower[0], 'is_quantized', False):
            model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))
        print(
            f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
        print(
            f"vision device: {model.model.vision_tower[0].device}, is_quantized: {getattr(model.model.vision_tower[0], 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
        self.model = model
        self.preprocessor['target'] = {'boxes': PlainBoxFormatter()}
        self.tokenizer = self.preprocessor['text']
        self.tokenizer.padding_side = 'left'
        self.gen_kwargs = dict(
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        if do_sample:
            self.gen_kwargs['top_p'] = 1.0
            self.gen_kwargs['temperature'] = float(1.0)
            self.gen_kwargs['do_sample'] = True
        else:
            self.gen_kwargs['do_sample'] = False

    @torch.no_grad()
    def do_generate(self, images, input_ids, attention_mask = None, max_new_tokens=128):
        output_ids = self.model.generate(
            images = images,
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            **self.gen_kwargs
        )
        input_token_len = input_ids.shape[-1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:])
        return [output.split('</s>')[0] for output in outputs]

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        ds = prepare_interactive(model_args, self.preprocessor)
        ds.set_image(get_image(image))
        ds.append_message(role=ds.roles[0], message = question, boxes = [], boxes_seq = [])
        model_inputs = ds.to_model_input()
        
        image = model_inputs['images'].to(torch.float16)
        input_text = model_inputs['input_text']
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt").to('cuda')
        outputs = self.do_generate(image, input_ids, max_new_tokens = max_new_tokens)
        return outputs[0]
        
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        input_image_list = []
        input_text_list = []
        for image, question in zip(image_list, question_list):
            ds = prepare_interactive(model_args, self.preprocessor)
            ds.set_image(get_image(image))
            ds.append_message(role=ds.roles[0], message = question, boxes = [], boxes_seq = [])
            model_inputs = ds.to_model_input()
            input_text_list.append(model_inputs['input_text'])
            model_inputs['images'] = model_inputs['images'].to(torch.float16)
            input_image_list.append(model_inputs['images'])
        input_image_list = torch.cat(input_image_list, dim = 0)
        input_dict = self.tokenizer(
            input_text_list, 
            padding="longest", return_length=True, 
            add_special_tokens=False, return_tensors="pt"
        ).to('cuda')
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        outputs = self.do_generate(input_image_list, input_ids, attention_mask, max_new_tokens)
        return outputs
    

    @torch.no_grad()
    def do_ppl(self, images, questions, answer_list, answer_pool, calib = False):
        images = torch.cat(images, dim = 0)
        input_dict = self.tokenizer(
            questions, 
            padding="longest", return_length=True, 
            add_special_tokens=False, return_tensors="pt"
        ).to('cuda')
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        answer_start_indices = []
        answer_end_indices = []
        template_token_list = []
        answer_token_list = []
        for template, option in zip(answer_list, answer_pool):
            template_token = self.tokenizer.encode(template, add_special_tokens=False)
            option_token = self.tokenizer.encode(option, add_special_tokens=False)
            template_token_list.append(torch.tensor(template_token))
            token_len = len(option_token)
            for index in range(len(template_token)):
                if template_token[index: index + token_len] == option_token:
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token)
                    break
            assert len(answer_start_indices) == len(template_token_list), "tokenizer encode answer in template different from answer only"
        
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
        )
        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        logits = logits[:, :-1]
        labels = input_ids[:, 1:]
        start_indices, end_indices = [], []
        for i in range(len(answer_list)):
            token_len = len(template_token_list[i])
            for index in range(labels.shape[1] - token_len, 0, -1):
                if torch.all(labels[i,index: index+token_len].cpu() == template_token_list[i]):
                    start_indices.append(index + answer_start_indices[i])
                    end_indices.append(index + answer_end_indices[i])
                    labels[i,:index] = -1
                    break
            assert len(start_indices) == (i+1), "tokenizer encode answer different from answer in conversation"

        loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), labels.reshape(-1),ignore_index=-1, reduction='none')
        loss = loss.reshape(-1, labels.shape[1]).float()
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
        input_image_list = []
        input_text_list = []
        for idx, (image, question) in enumerate(zip(image_list, question_list)):
            ds = prepare_interactive(model_args, self.preprocessor)
            ds.set_image(get_image(image))
            ds.append_message(role=ds.roles[0], message = question, boxes = [], boxes_seq = [])
            if CoT_list is not None:
                answer_list[idx] = CoT_list[idx] + '\n' + answer_list[idx]
            ds.append_message(role=ds.roles[1], message = answer_list[idx], boxes = [], boxes_seq = [])
            model_inputs = ds.to_model_input()
            input_text_list.append(model_inputs['input_text'])
            model_inputs['images'] = model_inputs['images'].to(torch.float16)
            input_image_list.append(model_inputs['images'])
        results = self.do_ppl(input_image_list, input_text_list, answer_list, answer_pool, calib=calib)
        return results

    def get_icl_prompt(self, question, ice, incontext_cfg):
        if incontext_cfg['use_pic']:
            raise NotImplementedError
        else:
            icl_question = ''
            for j in range(incontext_cfg['ice_num']):
                if not isinstance(ice[j]['gt_answers'], list):
                    icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers']}. "
                else:
                    icl_question += f"{ice[j]['question']}: {ice[j]['gt_answers'][0]}. "
            icl_question = icl_question + question
        return icl_question

    @torch.no_grad()
    def icl_batch_generate(self, image_list, question_list, ices, incontext_cfg, max_new_tokens=128):
        input_image_list = []
        input_text_list = []
        for image, question, ice in zip(image_list, question_list, ices):
            ds = prepare_interactive(model_args, self.preprocessor)
            ds.set_image(get_image(image))
            text = self.get_icl_prompt(question, ice, incontext_cfg)
            ds.append_message(role=ds.roles[0], message = text, boxes = [], boxes_seq = [])
            model_inputs = ds.to_model_input(incontext_cfg)
            input_text_list.append(model_inputs['input_text'])
            model_inputs['images'] = model_inputs['images'].to(torch.float16)
            input_image_list.append(model_inputs['images'])
        input_image_list = torch.cat(input_image_list, dim = 0)
        input_dict = self.tokenizer(
            input_text_list, 
            padding="longest", return_length=True, 
            add_special_tokens=False, return_tensors="pt"
        ).to('cuda')
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        outputs = self.do_generate(input_image_list, input_ids, attention_mask, max_new_tokens)
        return outputs, input_text_list

    @torch.no_grad()
    def icl_ppl_inference(self, image_list, question_list, answer_list, answer_options, ices, incontext_cfg, CoT_list=None):
        input_image_list = []
        input_text_list = []
        for idx, (image, question, ice) in enumerate(zip(image_list, question_list, ices)):
            ds = prepare_interactive(model_args, self.preprocessor)
            ds.set_image(get_image(image))
            text = self.get_icl_prompt(question, ice, incontext_cfg)
            ds.append_message(role=ds.roles[0], message = text, boxes = [], boxes_seq = [])
            if CoT_list is not None:
                answer_list[idx] = CoT_list[idx] + '\n' + answer_list[idx]
            ds.append_message(role=ds.roles[1], message = answer_list[idx], boxes = [], boxes_seq = [])
            model_inputs = ds.to_model_input(incontext_cfg)
            input_text_list.append(model_inputs['input_text'])
            model_inputs['images'] = model_inputs['images'].to(torch.float16)
            input_image_list.append(model_inputs['images'])
        results = self.get_ppl_results(input_text_list, input_image_list, answer_list, answer_options)
        return results, input_text_list

    def do_calibration(self, image_list, question_list, answer_list, answer_pool, CoT_list=None):
        return super().do_calibration(image_list, question_list, answer_list, answer_pool, CoT_list)
    
    