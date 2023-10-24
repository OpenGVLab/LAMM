import torch

class TestBase:

    def __init__(self, **kwargs) -> None:
        pass

    def move_to_device(self):
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def do_generate(self, images, questions, max_new_tokens):
        '''
            Direct generate answers with single image and questions, max_len(answer) = max_new_tokens
        '''
        return [''] * len(images) 

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens):
        '''
            process a single input image and instruction, and then do_generate
        '''
        return self.do_generate([image], [question], max_new_tokens=max_new_tokens)[0]

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens):
        '''
            process a batch of images and questions, and then do_generate
        '''
        return self.do_generate(image_list, question_list, max_new_tokens)
    
    @torch.no_grad()
    def do_ppl(self, images, questions, answer_list, answer_pool, calib = False):
        '''
            PPL generate answers with images and questions
            :param answer_list: list of answers with templates
            :param answer_pool: list of answers
            :param calib: output confidence for calibration evaluation
        '''
        return [0]*len(images)

    @torch.no_grad()
    def ppl_inference(self, image_list, question_list, answer_list, answer_pool, CoT_list = None, calib = False):
        '''
            process a batch of images and questions, and then do_ppl
            :param CoT_list: batch of CoT answers, the CoT is regarded as a part of ppl output
        '''
        return self.do_ppl(image_list, question_list, answer_list, answer_pool, calib=calib)

    @torch.no_grad()
    def icl_batch_generate(self, image_list, question_list, ices, incontext_cfg, max_new_tokens):
        '''
            process a batch of images and questions with ICE, and then do_generate
        '''
        return [''] * len(image_list) 

    @torch.no_grad()
    def icl_ppl_inference(self, image_list, question_list, answer_list, answer_pool, ices, incontext_cfg, CoT_list = None):
        '''
        
        '''
        return [0]*len(image_list)

    @torch.no_grad()
    def do_calibration(self, image_list, question_list, answer_list, answer_pool, CoT_list = None):
        return self.ppl_inference(image_list, question_list, answer_list, answer_pool, CoT_list = CoT_list, calib=True)
    


    
