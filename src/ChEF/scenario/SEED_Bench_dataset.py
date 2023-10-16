import os
import json
from torch.utils.data import Dataset

class SEEDBenchDataset(Dataset):
    task_name = 'VQA'
    dataset_name = 'SEEDBench'
    def __init__(self,
                 base_data_path,
                 ppl_cfg = True,
                 **kwargs
        ):
        self.base_data_path = base_data_path
        self.img_base_path = os.path.join(self.base_data_path, 'SEED-Bench-image')
        json_path = os.path.join(self.base_data_path, 'SEED-Bench.json')
        data = json.load(open(json_path, 'rb'))['questions']
        self.data = []
        for item in data:
            if item['data_type'] == 'image':
                self.data.append(item)
        self.ppl_cfg = ppl_cfg
        self.choices = 'ABCD'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        question = f'Question: {question}\nAnswer:'
        img_path = os.path.join(self.img_base_path, item['data_id'])
        gt_choices = [item['choice_a'], item['choice_b'], item['choice_c'], item['choice_d']]
        gt_choice = self.choices.index(item['answer'])
        gt_answers = gt_choices[gt_choice]
        id = str(item['question_id']) if 'question_id' in item else str(idx)
        res_dict = {
            'id': id,
            "image_path": img_path,
            "question": question,
            "gt_answers": gt_answers,
            "gt_choice": gt_choice,
            "gt_choices": gt_choices
        }
        if self.ppl_cfg:
            res_dict['options'] = gt_choices
        return res_dict
