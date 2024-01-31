import os
import json
import copy
from torch.utils.data import Dataset
import random
class Winoground_Cap_Dataset(Dataset):
    task_name = 'Winoground'
    dataset_name = 'Winoground_Cap'

    def __init__(self, base_data_path, **kwargs):
        self.base_data_path = base_data_path
        super().__init__()
        self.data = []  # 存储加载的数据
        
        with open(os.path.join(base_data_path, f'examples.jsonl'), 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                json_obj['question'] = f"Caption 1: {json_obj['caption_0']}.\nCaption 2: {json_obj['caption_1']}\n"
                json_obj['options'] = ['Caption 1 matches the Image 1', 
                                       'Caption 1 matches the Image 2',
                                        'Caption 2 matches the Image 1',
                                        'Caption 2 matches the Image 2'
                                      ]
                json_obj['main_id'] = json_obj['id']
                json_obj['image_0'] = os.path.join(base_data_path, 'images', json_obj['image_0'])
                json_obj['image_1'] = os.path.join(base_data_path, 'images', json_obj['image_1'])
                self.data.append(json_obj)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        res_dict = {
            'id': item['id'],
            'main_id': item['main_id'],
            'image_path': [item['image_0']+'.png', item['image_1']+'.png'],
            'question': item['question'],
            'gt_answers': ['Caption 1 matches the Image 1', 'Caption 2 matches the Image 2'],
            'options': item['options']
        }
        return res_dict

class WinogroundDataset(Dataset):
    task_name = 'Winoground'
    dataset_name = 'Winoground'

    def __init__(self, base_data_path, **kwargs):
        self.base_data_path = base_data_path
        super().__init__()
        self.data = []  # 存储加载的数据
        random.seed(2023)
        with open(os.path.join(base_data_path, f'examples.jsonl'), 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                
                json_obj['options'] = ['Yes', 'No']
                json_obj['main_id'] = json_obj['id']
                rd = random.randint(1,2)
                #import ipdb;ipdb.set_trace()
                if rd&1 > 0:
                    tmp = json_obj['caption_0']
                    json_obj['caption_0'] = json_obj['caption_1']
                    json_obj['caption_1'] = tmp
                json_obj['question'] = f"Caption 1: {json_obj['caption_0']}.\nCaption 2: {json_obj['caption_1']}\n"

                img0 = json_obj['image_0']
                img1 = json_obj['image_1']
                if rd&2 > 0:
                    json_obj['image_0'] = os.path.join(base_data_path, 'images', img1)
                    json_obj['image_1'] = os.path.join(base_data_path, 'images', img0)
                else:
                    json_obj['image_0'] = os.path.join(base_data_path, 'images', img0)
                    json_obj['image_1'] = os.path.join(base_data_path, 'images', img1)
                json_obj['shuffle'] = rd
                sub0 = copy.deepcopy(json_obj)
                sub1 = copy.deepcopy(json_obj)
                sub2 = copy.deepcopy(json_obj)
                sub3 = copy.deepcopy(json_obj)
                sub0['type'] = 'c0i0'
                sub1['type'] = 'c0i1'
                sub2['type'] = 'c1i0'
                sub3['type'] = 'c1i1'
                sub0['id'] = json_obj['main_id']*4
                sub1['id'] = json_obj['main_id']*4+1
                sub2['id'] = json_obj['main_id']*4+2
                sub3['id'] = json_obj['main_id']*4+3
                sub0['question'] += 'Does the Caption 1 match the Image 1?' #Answer yes or no.'
                sub1['question'] += 'Does the Caption 1 match the Image 2?' #Answer yes or no.'
                sub2['question'] += 'Does the Caption 2 match the Image 1?' #Answer yes or no.'
                sub3['question'] += 'Does the Caption 2 match the Image 2?' #Answer yes or no.'
                #sub0['question'] += 'Does Caption 1 and Image 1 correspond to each other?'
                #sub1['question'] += 'Does Caption 1 and Image 2 correspond to each other?'
                #sub2['question'] += 'Does Caption 2 and Image 1 correspond to each other?'
                #sub3['question'] += 'Does Caption 2 and Image 2 correspond to each other?'
                sub0['gt_answers'] = 'Yes'
                sub1['gt_answers'] = 'No'
                sub2['gt_answers'] = 'No'
                sub3['gt_answers'] = 'Yes'
                self.data = self.data + [sub0, sub1, sub2, sub3]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        res_dict = {
            'id': item['id'],
            'main_id': item['main_id'],
            'image_path': [item['image_0']+'.png', item['image_1']+'.png'],
            'question': item['question'],
            'gt_answers': item['gt_answers'],
            'options': ['Yes', 'No'],
            'type': item['type'],
            'shuffle':item['shuffle']
        }
        return res_dict