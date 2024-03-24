import torch
import torch.nn.functional as F
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from .mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from .mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from .utils import Conversation, SeparatorStyle
from .test_base import TestBase
import numpy as np
prompt_template_multi = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

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
    def __init__(self, device, model_path, **kwargs):
        self.model = MplugOwlForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.device = device
        self.move_to_device(device)
        self.model.eval()

    def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None, batch_answers=None, **kwargs):
        lenimg = 1 if isinstance(image_list,str) else len(image_list)
        prompt = prompt_template_multi + "\n".join(["Human: <image>"] * lenimg) + f"\nHuman: {prompt}\nAI:"
        if CoT_answer_list is not None:
            prompt += CoT_answer_list[idx]
        if batch_answers is not None:
            prompt += ' ' + batch_answers[idx]
        return prompt
    
    def build_input_image(self, image_list):
        images = self.get_image_list(image_list)
        images = [self.image_processor(image, return_tensors='pt').pixel_values for image in images]
        images = torch.cat(images, dim=0).to(self.device, dtype=self.dtype)
        return images

    def do_generate(self, image_list: list, prompt: str, max_new_tokens, **kwargs):
        inputs = self.processor(text=[prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs['pixel_values'] = image_list # multi_len*bs, c, h, w
        generate_kwargs = {
            'do_sample': False,
            'top_k': 5,
            'max_length': max_new_tokens
        }
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in res.tolist()]
        return outputs[0]

    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        inputs = self.processor(text=batch_prompt)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        batch_images = torch.cat(batch_images, dim=0).to(self.device, dtype=self.dtype) # multi_len * bs, c, h, w
        inputs["pixel_values"] = batch_images
        labels = inputs['input_ids'].clone()[:,1:]
        outputs = self.model.generate(**inputs, ppl=True)
        
        logits = outputs['logits'][:,:-1].float()
        batch_option_ids = []
        for option in batch_options:
            batch_option_ids.append(self.tokenizer.encode(f'{option}', add_special_tokens=False, return_tensors='pt').squeeze(0)) 
        results = []
        for idx in range(labels.shape[0]):
            option_len = len(batch_option_ids[idx])
            non_zero_indices = torch.nonzero(labels[idx]!=-100, as_tuple=False).squeeze()
            start_index = non_zero_indices.max() - option_len + 1
            end_index = start_index + option_len
            if not np.all(labels[idx][start_index: end_index].detach().cpu().numpy() == batch_option_ids[idx].numpy()):
                import ipdb;ipdb.set_trace()
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean().item()
            results.append(score)
        return results