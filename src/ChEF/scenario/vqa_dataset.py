import os
import json
from torch.utils.data import Dataset
import random
#from data_process.augmentations.mild_mix_perturbation import MildMixPerturbation

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
def get_options(choices, option_content):
    option_list = []
    for idx, answer in enumerate(choices):
        optionstr = OPTION[idx]
        if option_content:
            option_list.append(f'({optionstr}) {answer}')
        else:
            option_list.append(f'({optionstr}')
    return option_list

def clean_question(question, generative = False): # delete context
    qlist = question.split('Options:')
    q = qlist[0].split('Context:')
    if not generative:
        res = 'Question: ' + q[0] + 'Options:' + qlist[1] + "\n"
    else:
        res = 'Question: ' + q[0] + "\n"
    
    return res

class ScienceQADataset(Dataset):
    """Example:
        data['question'] = "Question: What is the name of the colony shown?\nOptions: (A) Maryland (B) New Hampshire (C) Rhode Island (D) Vermont\n"
        data['options'] = ['(A', '(B', '(C', '(D']
    """
    task_name = 'VQA'
    dataset_name = 'ScienceQA'

    def __init__(self, base_data_path, ppl = False, option_content = False, option_map=None, img_crp=False, text_crp=False,split='31', generative = False, 
                  data_c_path = 'data/datasets/ChEF/ScienceQA_C', **kwargs):
        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', f'{self.task_name}_{self.dataset_name}.json')
        if text_crp:
            json_path = os.path.join(data_c_path, 'VQA_ScienceQA_C.json')
        self.data = json.load(open(json_path, 'rb'))
        self.ppl = ppl  # if true, return positive option and negative options 
        self.option_content = option_content # if true, return [(A) xxx]  instead of (A)
        self.map_type = None
        if option_map!=None:
            self.map_type = option_map['type']
            self.map_id = option_map['ids']
            '''
            data_tmp = []
            sub_idx = json.load(open(option_map['sub_idx'], 'rb'))
            for i in sub_idx:
                data_tmp.append(self.data[i])
            perc = option_map['perc']
            N = len(data_tmp)
            self.data=data_tmp[:int(N*perc)]
            '''
            if self.map_type!='unnatural':
                self.option_map=OPTION_MAP[self.map_type][option_map['ids']]
        self.data_c_path = data_c_path
        if img_crp:
            self.base_data_path = self.data_c_path
        self.img_crp = img_crp
        self.generative = generative
        #self.random_generator = random.Random()
        #self.random_generator.seed(2023)
        #self.mix_perb = MildMixPerturbation()

    def __len__(self):
        return len(self.data)

    def clean_crp_question(self,question):
        qlist = question.split('Options:')
        q = qlist[0].split('Context:')
        q[0] = self.mix_perb.perturb(q[0] ,self.random_generator) 
        return 'Question: ' + q[0] + 'Options:' + qlist[1] + "\n"
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['query']
        question = clean_question(question, self.generative)
        img_path = os.path.join(self.base_data_path,item['image'])
        gt_choice = item['gt_choice']
        gt_answers = item['gt_choices'][gt_choice]
        gt_choices = item['gt_choices']
        
        id = str(item['id']) if 'id' in item else str(idx)
        res_dict = {
            'id': id,
            "image_path": img_path,
            "question": question,
            "gt_answers": gt_answers,
            "gt_choice": gt_choice,
            "gt_choices": gt_choices
        }

        if self.generative:
            res_dict['options'] = gt_choices
            res_dict['gt_answers'] = gt_choices[res_dict['gt_choice']]
        else:
            res_dict['options'] = get_options(gt_choices, self.option_content)
            res_dict['gt_answers'] = '(' + OPTION[res_dict['gt_choice']] + ')'
        
        # res_dict['options'] = get_options(gt_choices, self.option_content)
        
        if self.map_type!=None:
            map_text = ''
            map_template='If the answer is "{}", you need to output "{}". '
            if self.map_type=='unnatural':
                if self.map_id==0:
                    option_map = res_dict['options'][1:]+res_dict['options'][:1]
                else:
                    option_map = res_dict['options'][-1:]+res_dict['options'][:-1]
            else:
                option_map = self.option_map
            
            
            for opid,opt in enumerate(res_dict['options']):
                map_text+=map_template.format(opt+')', option_map[opid])
            #map_text+='\n'
            res_dict['question']+=map_text
            res_dict['options']=option_map[:len(res_dict['options'])]
            #res_dict['CHOICES']=res_dict['options']
        #import ipdb;ipdb.set_trace()

        return res_dict


class ScienceQADataset_C(Dataset):
    """Example:
        data['question'] = "Question: What is the name of the colony shown?\nOptions: (A) Maryland (B) New Hampshire (C) Rhode Island (D) Vermont\n"
        data['options'] = ['(A)', '(B)', '(C)', '(D)']
    """
    task_name = 'VQA'
    dataset_name = 'ScienceQA'

    def __init__(self, base_data_path, ppl = False, option_content = True, option_map=None):
        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', f'{self.task_name}_{self.datasat_name}.json')
        self.data = json.load(open(json_path, 'rb'))
        self.ppl = ppl  # if true, return positive option and negative options 
        self.option_content = option_content # if true, return [(A) xxx]  instead of (A)
        self.map_type = None
        if option_map!=None:
            self.map_type = option_map['type']
            self.map_id = option_map['ids']
            data_tmp = []
            sub_idx = json.load(open(option_map['sub_idx'], 'rb'))
            for i in sub_idx:
                data_tmp.append(self.data[i])
            perc = option_map['perc']
            N =  len(data_tmp)
            self.data=data_tmp[:int(N*perc)]
            if self.map_type!='unnatural':
                self.option_map=OPTION_MAP[self.map_type][option_map['ids']]
        self.data_c_path = '/ssd/home/wangzhipin/data/sqa_c/imgs'
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['query']
        question = clean_question(question)
        img_path = os.path.join(self.data_c_path,item['image'])
        gt_choice = item['gt_choice']
        gt_answers = item['gt_choices'][gt_choice]
        gt_choices = item['gt_choices']
        
        id = str(item['id']) if 'id' in item else str(idx)
        res_dict = {
            'id': id,
            "image_path": img_path,
            "question": question,
            "gt_answers": gt_answers,
            "gt_choice": gt_choice,
            "gt_choices": gt_choices
        }
        
        res_dict['options'] = get_options(gt_choices, self.option_content)
        
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
            
            
            for opid,opt in enumerate(res_dict['options']):
                map_text+=map_template.format(opt+')', option_map[opid])
            #map_text+='\n'
            res_dict['question']+=map_text
            res_dict['options']=option_map[:len(res_dict['options'])]
            res_dict['CHOICES']=res_dict['options']
        #import ipdb;ipdb.set_trace()
        return res_dict


if __name__ == '__main__':
    dataset = ScienceQADataset(base_data_path='data/datasets/LAMM/2D_Benchmark', ppl=True, generative=True)
    data = dataset[0]
    import ipdb;ipdb.set_trace()