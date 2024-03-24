import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import numpy as np
from .otter.modeling_otter import OtterForConditionalGeneration
from .instruct_blip.models.eva_vit import convert_weights_to_fp16
from .utils import Conversation, SeparatorStyle
from .test_base import TestBase


CONV_VISION = Conversation(
    system='',
    roles=("<image>User", "GPT"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep=" ",
)

class TestOtter(TestBase):
    def __init__(self, model_path, device='cuda', **kwargs) -> None:
        self.model = OtterForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()
        self.tokenizer.padding_side = "left"
        self.move_to_device(device)

    def move_to_device(self, device):
        self.dtype = torch.float16
        self.device = device
        convert_weights_to_fp16(self.model.vision_encoder)
        self.model = self.model.to(self.device, dtype=self.dtype)

    def build_input_image(self, image_list):
        imgs = self.get_image_list(image_list)
        imgs = self.image_processor.preprocess(imgs, return_tensors="pt")["pixel_values"].unsqueeze(1) # chunk * 1 * c h w
        return imgs
    
    def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None, batch_answers=None, **kwargs):
        lenimg = 1 if isinstance(image_list,str) else len(image_list)
        prompt = " ".join(["<image>"] * lenimg) + f" User: {prompt} GPT: <answer>"
        if CoT_answer_list is not None:
            prompt += ' ' + CoT_answer_list[idx] + '\n'
        if batch_answers is not None:
            prompt += ' ' + batch_answers[idx]
        return prompt
    
    def do_generate(self, image_list: torch.Tensor, prompt: str, max_new_tokens, **kwargs):
        vision_x=torch.stack([image_list], dim=0)
        vision_x=vision_x.to(self.model.device, dtype=self.dtype)
        lang_x = self.model.text_tokenizer([prompt], return_tensors="pt")
        generated_text = self.model.generate(
            vision_x=vision_x,
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

    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        vision_x = torch.stack(batch_images, dim=0).to(self.model.device, dtype=self.dtype)
        lang_x = self.model.text_tokenizer(batch_prompt, return_tensors="pt", padding=True)
        input_ids=lang_x["input_ids"].to(self.model.device)
        attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype)
        output = self.model(vision_x=vision_x, lang_x=input_ids, attention_mask=attention_mask)
        logits = output['logits'][:,:-1].float()
        labels = input_ids[:,1:]

        batch_option_ids = []
        for option in batch_options:
            batch_option_ids.append(self.model.text_tokenizer(option, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)) 
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
    