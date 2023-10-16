import os
import json
from torch.utils.data import Dataset
import random
class POPE_COCO_Random_Dataset(Dataset):
    task_name = 'POPE'
    dataset_name = 'POPE_COCO_random'
    def __init__(self, base_data_path):
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,f'coco_pope_random.json')
        self.data = [json.loads(q) for q in open(json_path, 'r')]
        #self.data = [self.data[54]]*10
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['question_id']) if 'question_id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': os.path.join(self.base_data_path,'val2014', self.data[index]['image']),
            'question': self.data[index]['text'],
            'gt_answers': self.data[index]['label'],
            'options': ['Yes', 'No'],
        }

        return res_dict

class POPE_COCO_Popular_Dataset(Dataset):
    task_name = 'POPE'
    dataset_name = 'POPE_COCO_popular'
    def __init__(self, base_data_path):
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,f'coco_pope_popular.json')
        self.data = [json.loads(q) for q in open(json_path, 'r')]
        #self.data = [self.data[15]]*10
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['question_id']) if 'question_id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': os.path.join(self.base_data_path,'val2014', self.data[index]['image']),
            'question': self.data[index]['text'],
            'gt_answers': self.data[index]['label'],
            'options': ['Yes', 'No'],
        }

        return res_dict

class POPE_COCO_Adversarial_Dataset(Dataset):
    task_name = 'POPE'
    dataset_name = 'POPE_COCO_adversarial'
    def __init__(self, base_data_path):
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,f'coco_pope_adversarial.json')
        self.data = [json.loads(q) for q in open(json_path, 'r')]
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['question_id']) if 'question_id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': os.path.join(self.base_data_path,'val2014', self.data[index]['image']),
            'question': self.data[index]['text'],
            'gt_answers': self.data[index]['label'],
            'options': ['Yes', 'No'],
        }
        return res_dict