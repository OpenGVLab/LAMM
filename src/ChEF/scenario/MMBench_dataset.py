import base64
import io
import random
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import json
OPTION=['A','B','C','D','E','F','G','H']
OPTION_MAP = {'natural':[['1','2','3','4','5','6','7','8'],
                          ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'],
                          ['first','second', 'third', 'fourth', 'fifth','sixth'],
                          ['(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)'],
                         ['α','β','γ','δ','ε','ζ','η','θ']],
             'neutral':[
                 ["Smith", "Johnson", "Williams", "Jones", "Brown","Davis", "Miller", "Wilson"],
                 ["foo", "dog", "hip", "oh",'cat','lake','river','joy'],
                 ['~','@','#','$', '%','^','&','*'],
                 
                ]
}

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMBenchDataset(Dataset):
    """
        example: 
        data['question'] = "Question: Which statement describes the Great Victoria Desert ecosystem?\nOptions: (A) It has thick, moist soil. (B) It has dry, thin soil.\n"
        data['options'] = ['(A)', '(B)']
    """
    task_name = 'VQA'
    dataset_name = 'MMBench'
    def __init__(self,
                 base_data_path,
                 split = 'dev',
                 sys_prompt='There are several options:',
                 hint = True, 
                 ppl_cfg = None,
                 option_map=None,
                 text_crp=False,
                 img_crp=False,
                 generative=False,
                 data_c_path = 'data/datasets/ChEF/MMBench_C',
                 **kwargs
        ):
        self.df = pd.read_csv(os.path.join(base_data_path, f'mmbench_{split}_20230712.tsv'), sep='\t')
        self.ppl_cfg = ppl_cfg
        self.sys_prompts = sys_prompt
        self.hint = hint
        if self.ppl_cfg:
            self.content_only = self.ppl_cfg.get('content_only', False)
        self.circularidx = []
        for i in range(len(self.df)):
            index = self.df.iloc[i]['index']
            if index > 1e6:
                self.circularidx.append(i)
                
        self.map_type = None
        if option_map!=None:
            self.map_type = option_map['type']
            self.map_id = option_map['ids']
            if self.map_type!='unnatural':
                self.option_map=OPTION_MAP[self.map_type][option_map['ids']]
        self.text_crp = text_crp
        self.img_crp = img_crp
        self.img_c_path = os.path.join(data_c_path, 'images')
        self.txt_c = json.load(open(os.path.join(data_c_path, 'MMBench_C.json'), 'rb'))
        self.generative = generative

        self.data = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        if self.img_crp:
            image=os.path.join(self.img_c_path,f'{str(idx)}.png')
        else:
            image = decode_base64_to_image(image)
        question = self.df.iloc[idx]['question']
        hint = self.load_from_df(idx, 'hint')
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompts}\n'
        for key, item in options.items():
            options_prompt += f'({key}) {item}\n'
        
        if self.generative:
            options_prompt = '' # no options

        if hint is not None and self.hint:
            question = f'{hint} {question} {options_prompt}'
        else:
            question = f'{question} {options_prompt}'
        
        
        data = {
            'id': str(index),
            'image_path': image,
            'question': question,
            'gt_choices': [value for value in options.values()]
        }
        
        data['gt_choice'] = option_candidate.index(answer) if answer is not None else None
        data['gt_answers'] = options[answer] if answer is not None else None
        if self.text_crp:
            data['question'] = self.txt_c[idx]["query"]
            data['gt_choices'] = self.txt_c[idx]["gt_choices"]
            data['gt_choice'] = self.txt_c[idx]["gt_choice"]
            data['gt_answers'] = data['gt_answers'][data['gt_choice']]
            
            options = {
                option_candidate[idx]: choice for idx,choice in enumerate(data['gt_choices'])
            }
            for op in option_candidate:
                data['question']=data['question'].replace(f' ({op})', f'\n({op})')
        if self.ppl_cfg:
            option_list = []
            for key, item in options.items():
                option_list.append(f'{item}' if (self.content_only or self.generative) else  f'({key}')
            data['options'] = option_list

            data['gt_answers'] = '(' + option_candidate[data['gt_choice']] + ')'

            if self.map_type!=None:
                map_text = ''
                map_template='If the answer is "{}", you need to output "{}". '
                if self.map_type=='unnatural':
                    if self.map_id==0:
                        option_map = data['options'][1:]+data['options'][:1]
                    else:
                        option_map = data['options'][-1:]+data['options'][:-1]
                else:
                    option_map = self.option_map


                for opid,opt in enumerate(data['options']):
                    map_text+=map_template.format(opt+')', option_map[opid])

                data['question']+=map_text
                data['options']=option_map[:len(data['options'])]

        return data
    
    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None

if __name__ == '__main__':
    dataset = MMBenchDataset(base_data_path='data/datasets/MMBench', split='dev', hint=True, ppl_cfg = dict(content_only = False), generative=True)
    data = dataset[0]