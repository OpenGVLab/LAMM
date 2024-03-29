import torch
from .utils import get_multi_imgs
from PIL import Image
class TestBase:

    def __init__(self, **kwargs) -> None:
        pass

    def move_to_device(self, device):
        if device is not None:
            self.device = device
            self.dtype = torch.float16
            self.model = self.model.to(self.device, dtype=self.dtype)
            return 
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def do_generate(
        self, 
        image_list: list, # num_image * image 
        prompt: str, 
        max_new_tokens,
        **kwargs
    ):
        '''
            Direct generate answers with images and questions, max_len(answer) = max_new_tokens
        '''
        raise NotImplementedError
    
    def build_conversation(
        self, 
        idx, 
        image_list, 
        prompt, 
        CoT_answer_list=None, 
        batch_answers=None,
        **kwargs
    ):
        raise NotImplementedError

    def get_image_list(self, image_list): # [multi_image_len]
        if not isinstance(image_list, list):
            image_list = [image_list]
        return get_multi_imgs(image_list)

    def build_input_image(self, image_list):
        raise NotImplementedError

    @torch.no_grad()
    def batch_generate(self, batch_images, batch_prompt, max_new_tokens, **kwargs):
        outputs = []
        for idx, (image_list, prompt) in enumerate(zip(batch_images, batch_prompt)):
            input_prompt = self.build_conversation(idx, image_list, prompt, generate=True, **kwargs)
            input_image_list = self.build_input_image(image_list)
            output = self.do_generate(input_image_list, input_prompt, max_new_tokens)
            outputs.append(output)
        return outputs
    
    @torch.no_grad()
    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def ppl_inference(
        self, 
        batch_images, 
        batch_prompt, 
        batch_options,
        **kwargs, 
    ):
        '''
            process a batch of images and questions, and then do_ppl
        '''
        input_images, input_prompts = [], []
        for idx, (image_list, prompt) in \
                enumerate(zip(batch_images, batch_prompt)):
            input_prompt = self.build_conversation(idx, image_list, prompt, generate=False, **kwargs)
            input_image_list = self.build_input_image(image_list)
            input_prompts.append(input_prompt)
            input_images.append(input_image_list)
        return self.do_ppl(input_images, input_prompts, batch_options, **kwargs)
    
    def horizontal_concat(self, image_list):
        total_width = sum(img.width for img in image_list)
        max_height = max(img.height for img in image_list)
        dst = Image.new('RGB', (total_width, max_height))
        current_width = 0
        for img in image_list:
            dst.paste(img, (current_width, 0))
            current_width += img.width
        return dst
