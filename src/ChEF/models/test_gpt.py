
from openai import OpenAI
import base64
from .test_base import TestBase
import copy
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_message(image_list, prompts):
    msg = {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompts,
        },
      ],
    }
    img_template = {
          "type": "image_url",
          "image_url": {
           "url": "",
          }
        }
    url_template = "data:image/jpeg;base64,{}"
    for image in image_list:
        img_msg = copy.deepcopy(img_template)
        base64_image = encode_image(image)
        #encode_image(image_list[0]) == encode_image(image_list[1])
        img_msg['image_url']['url'] = url_template.format(base64_image)
        msg['content'].append(img_msg)
    return [msg]
    
class TestGPT(TestBase):
    def __init__(self, api_key="", gpt_name = "gpt-4-vision-preview", **kwargs) -> None:
        self.client = OpenAI(api_key=api_key)
        self.gpt_name = gpt_name

    def batch_generate(self, image_list, question_list, max_new_tokens, **kwargs):
        '''
            process a batch of images and questions, and then do_generate
        '''
        answers = []
        for imgs, prompts in zip(image_list, question_list):
            if len(imgs)>1:
              prompts+=' The order in which I upload images is the order of the images.'
            msg = create_message(imgs, prompts)
            
            try:
              response = self.client.chat.completions.create(
                model=self.gpt_name,
                messages=msg,
                max_tokens=max_new_tokens,
              )
            except Exception as e:
              error_msg = str(e).split('message')[1][4:].split('\'')[0]
              answers.append("##Error##:" + error_msg)
              continue
            answers.append(response.choices[0].message.content)
            
        '''
            Direct generate answers with single image and questions, max_len(answer) = max_new_tokens
        '''
        return answers
       #len(msg[0]['content'][3]['image_url']['url'])