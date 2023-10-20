import torch
from .minigpt4.common.config import Config
from .minigpt4.common.registry import registry
from .minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from .minigpt4.models import *
from .minigpt4.processors import *
from .test_base import TestBase


from .utils import get_image


class TestMiniGPT4(TestBase):
    def __init__(self, 
                 model_path,
                 cfg_path = 'models/minigpt4/minigpt4_eval.yaml',
                 **kwargs
                 ):
        cfg = Config(cfg_path, model_path)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cpu')
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model, self.vis_processor = model, vis_processor
        self.model.llama_model = self.model.llama_model.float().to('cpu')
        self.chat = Chat(model, vis_processor, device='cpu')
        self.ice_imgs_emb = None
        self.move_to_device()

    def move_to_device(self):
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
            self.chat.device = 'cuda'
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.chat.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.chat.move_stopping_criteria_device(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        chat_state = CONV_VISION.copy()
        img_list = []
        if image is not None:
            image = get_image(image)
            self.chat.upload_img(image, chat_state, img_list)
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=max_new_tokens)[0]

        return llm_message

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        image_list = [get_image(image) for image in image_list]
        chat_list = [CONV_VISION.copy() for _ in range(len(image_list))]
        batch_outputs = self.chat.batch_answer(image_list, question_list, chat_list, max_new_tokens=max_new_tokens)
        return batch_outputs
    
    @torch.no_grad()
    def ppl_inference(self, image_list, question_list, answer_list, answer_options, CoT_list = None, calib = False):
        image_list = [get_image(image) for image in image_list]
        chat_list = [CONV_VISION.copy() for _ in range(len(image_list))]
        ppl_results = self.chat.ppl_answer(image_list, question_list, chat_list, answer_list, answer_options, CoT_list=CoT_list, calib=calib)
        return ppl_results

    @torch.no_grad()
    def icl_batch_generate(self, image_list, question_list, ices, incontext_cfg, max_new_tokens=128):
        sample_data = ices
        ice_imgs_emb = self.get_ice_imgs_emb(sample_data, incontext_cfg)
        if incontext_cfg['use_pic'] and incontext_cfg['retriever_type'] == 'fixed':
            ice_imgs_emb = [ice_imgs_emb for _ in image_list]
        chat_list = [CONV_VISION.copy() for _ in range(len(image_list))]
        batch_outputs, prompts = self.chat.icl_batch_answer(image_list, question_list, chat_list, ice_imgs_emb, sample_data, incontext_cfg, max_new_tokens=max_new_tokens)
        return batch_outputs, prompts

    @torch.no_grad()
    def cali_inference(self, image_list, question_list, answer_list, answer_options, CoT_list = None):
        image_list = [get_image(image) for image in image_list]
        chat_list = [CONV_VISION.copy() for _ in range(len(image_list))]
        ppl_results = self.chat.cali_answer(image_list, question_list, chat_list, answer_list,answer_options, CoT_list)
        return ppl_results 

    @torch.no_grad()
    def get_ice_imgs_emb(self, ices, incontext_cfg):
        if incontext_cfg['retriever_type'] != 'fixed' and incontext_cfg['use_pic']:
            ices_images = []
            for ice in ices:
                ice_images = []
                for i in range(len(ice)):
                    ice_images.append(ice[i]['image_path'])
                ices_images.append(ice_images)
            self.ice_imgs_emb = [self.chat.get_imgs_emb(ice_img) for ice_img in ices_images]
        elif incontext_cfg['use_pic'] and self.ice_imgs_emb is None:
            ice_images = [ice['image_path'] for ice in ices[0]]
            self.ice_imgs_emb = self.chat.get_imgs_emb(ice_images)
        return self.ice_imgs_emb

    @torch.no_grad()
    def icl_ppl_inference(self, image_list, question_list, answer_list, answer_options, ices, incontext_cfg, CoT_list=None):
        sample_data = ices
        ice_imgs_emb = self.get_ice_imgs_emb(sample_data, incontext_cfg)
        if incontext_cfg['use_pic'] and incontext_cfg['retriever_type'] == 'fixed':
            ice_imgs_emb = [ice_imgs_emb for _ in image_list]
        image_list = [get_image(image) for image in image_list]
        chat_list = [CONV_VISION.copy() for _ in range(len(image_list))]
        batch_outputs, prompts = self.chat.icl_ppl_batch_answer(image_list, question_list, chat_list, answer_list, answer_options, ice_imgs_emb, sample_data, incontext_cfg, CoT_list)
        return batch_outputs, prompts
    
    @torch.no_grad()
    def do_calibration(self, image_list, question_list, answer_list, answer_pool, CoT_list=None):
        return super().do_calibration(image_list, question_list, answer_list, answer_pool, CoT_list)