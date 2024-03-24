import os
import json
import yaml
import re
import sys
#solve package Duplicate name
lib_paths = [path for path in sys.path if 'LAMM/src' not in path ]
chef_paths = [path for path in sys.path if 'LAMM/src' in path ]
sys.path = lib_paths + chef_paths
from datasets import load_dataset, concatenate_datasets

DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}


CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}



def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches

def process_single_sample(data, base_data_path):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    #if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
    #    return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
    #         'image': None, 'question_type': data['question_type']}
    #else:
    images = []
    for i in range(1,8):
        key = f'image_{i}'
        if key not in data or data[key]==None:
            continue
        path = os.path.join(base_data_path, data['id']+f'_{str(i-1)}.png')
        images.append(path)
    
    return {'id': data['id'], 'question': question, 'options': data['options'], 'gt_choice': data['answer'],
             'image_path': images, 'question_type': data['question_type']}



config = dict(
    task_instructions="",
    multi_choice_example_format="{}\n{}\nAnswer with the option's letter from the given choices directly.",
    short_ans_example_format="{}\nAnswer the question using a single word or phrase.")


# DATA PROCESSING
def construct_prompt(sample):
    question = sample['question']
    options = eval(sample['options'])
    example = ""
    if sample['question_type'] == 'multiple-choice':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict.update(sample)
        #res_dict['index2ans'] = index2ans
        #res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt

        res_dict['gt_answers'] = options[ord(sample['gt_choice'].upper()) - ord('A')]
    else:
        empty_prompt_sample_structure = config['short_ans_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict.update(sample)
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_answers'] = sample['gt_choice']
    res_dict['options'] = options
    res_dict['question'] = res_dict['final_input_prompt']
    res_dict['source'] = 'MMMU'
    subject_words = sample['id'].split('_')[1:-1]
    subject_words = '_'.join(subject_words)
    res_dict['subject'] = subject_words
    return res_dict




import os
import json
from torch.utils.data import Dataset

class MMMUDataset(Dataset):
    task_name = 'MMMU'
    dataset_name = 'MMMU'
    def __init__(self, base_data_path, img_folder, ppl=False, **kwargs):
        self.base_data_path = base_data_path
        self.img_folder = img_folder
        super().__init__()
        self.ppl = ppl
        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(base_data_path, subject, split = 'validation')
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        self.data = concatenate_datasets(sub_dataset_list)
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = process_single_sample(self.data[int(index)], self.img_folder)
        item = construct_prompt(item)
        return item
