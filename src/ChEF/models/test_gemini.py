import google.generativeai as genai
from time import sleep
# Used to securely store your API key
# from google.colab import userdata
from .test_base import TestBase
from IPython.display import display
from IPython.display import Markdown
from IPython.display import Image
from IPython.core.display import HTML
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
class TestGemini(TestBase):
    def __init__(self, api_key="", gemini_name = "gemini-pro-vision", safety_block_none=False, **kwargs) -> None:
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(gemini_name)
        self.gemini_name = gemini_name
        self.safety_block_none = safety_block_none

    def batch_generate(self, image_list, question_list, max_new_tokens, **kwargs):
        '''
            process a batch of images and questions, and then do_generate
        '''
        answers = []
        generation_config=genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens=max_new_tokens,
        temperature=1.0)
        for imgs, prompts in zip(image_list, question_list):
            img_list = [Image(img) for img in imgs]
            try_time = 0
            response = None
            while True:
                try:
                    if self.safety_block_none:
                        response = self.model.generate_content(contents= [prompts] + img_list, safety_settings=safety_settings, generation_config = generation_config)
                    else:
                        response = self.model.generate_content(contents= [prompts] + img_list, generation_config = generation_config)
                    response.resolve()
                    answers.append(str(response.text).strip())
                    try_time = 0
                    break
                except Exception as e:
                    if try_time>=5:
                        try:
                            answers.append("##ERROR## "+str(response.prompt_feedback).strip())
                        except:
                            answers.append("##ERROR##")
                        break
                    try_time += 1
                    continue
        '''
            Direct generate answers with single image and questions, max_len(answer) = max_new_tokens
        '''
        return answers
       