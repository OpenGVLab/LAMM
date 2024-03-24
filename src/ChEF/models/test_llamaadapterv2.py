import torch
import torch.nn.functional as F

from . import llama_adapter_v2
from .utils import *
from .test_base import TestBase

CONV_VISION = Conversation(
    system='Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n',
    roles=("Instruction", "Input", "Response"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="### ",
)
class TestLLamaAdapterV2(TestBase):
    def __init__(
        self,
        model_path, 
        device='cuda',        
        max_seq_len = 1024,
        max_batch_size = 40,
        **kwargs) -> None:
        llama_dir = model_path
        model, preprocess = llama_adapter_v2.load("LORA-BIAS-7B", llama_dir, download_root=llama_dir, max_seq_len=max_seq_len, max_batch_size=max_batch_size, device='cpu')
        self.img_transform = preprocess
        self.model = model.eval()
        self.tokenizer = self.model.tokenizer
        self.move_to_device(device)

    def build_input_image(self, image_list):
        image_list = self.get_image_list(image_list)
        if len(image_list) == 1:
            image = image_list[0]
        else:
            image = self.horizontal_concat(image_list) 
        image = np.array(image)[:, :, ::-1]
        image = Image.fromarray(np.uint8(image))
        img = self.img_transform(image)
        return img
    
    def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None, batch_answers=None, generate=True, **kwargs):
        prompt = llama_adapter_v2.format_prompt(prompt)
        if CoT_answer_list is not None:
            prompt += ' ' + CoT_answer_list[idx] + '\n'
        if generate:
            return prompt
        else:
            return prompt, batch_answers[idx]

    def do_generate(self, image_list: torch.Tensor, prompt: str, max_new_tokens, **kwargs):
        imgs = image_list.unsqueeze(0).to(self.device)
        prompts = [prompt]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens, device=self.device)
        result = results[0].strip()
        return result
    
    
    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        prompts = [item[0] for item in batch_prompt]
        batch_answers = [item[1] for item in batch_prompt]
        images = torch.stack(batch_images, dim=0).to(self.device)
        logits, labels = self.model.ppl_generate(images, prompts, batch_answers, device=self.device)
        logits = logits.float()
        batch_option_ids = []
        for option in batch_options:
            batch_option_ids.append(self.tokenizer.encode(option, bos=False, eos=False)) 
        results = []
        for idx in range(labels.shape[0]):
            option_len = len(batch_option_ids[idx])
            non_zero_indices = torch.nonzero(labels[idx]!=-100, as_tuple=False).squeeze()
            start_index = non_zero_indices.max() - option_len + 1
            end_index = start_index + option_len
            if not np.all(labels[idx][start_index: end_index].detach().cpu().numpy() == np.array(batch_option_ids[idx])):
                import ipdb;ipdb.set_trace()
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean().item()
            results.append(score)
        return results
