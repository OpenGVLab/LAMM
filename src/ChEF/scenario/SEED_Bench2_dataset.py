import os
import json
from torch.utils.data import Dataset


class SEEDBench2Dataset(Dataset):
    task_name = 'VQA'
    dataset_name = 'SEEDBench2'
    def __init__(self,
                 base_data_path,
                 ppl_cfg = True,
                 **kwargs
        ):
        self.base_data_path = base_data_path
        self.img_base_path = os.path.join(self.base_data_path, 'image/SEED-Bench-2-image')
        self.cc3m_dir = os.path.join(self.base_data_path, 'image/cc3m-image')
        json_path = os.path.join(self.base_data_path, 'SEED-Bench_v2_level1_2_3.json')
        data = json.load(open(json_path, 'rb'))['questions']
        self.data = []
        for item in data:
            #'Multiple Images', 'Interleaved Image', 'Image & Text Generation', 'Image Generation', 'Video', 'Single Image'
            if item['data_type'] != 'Image Generation':
                self.data.append(item)
        self.ppl_cfg = ppl_cfg
        self.choices = 'ABCD'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        question = f'Question: {question}\nAnswer:'
        if item['data_source'] == 'cc3m':
            if type(item['data_id']) is list:
                img_path = [os.path.join(self.cc3m_dir, path) for path in item['data_id']]
            else:
                img_path = os.path.join(self.cc3m_dir, item['data_id'])
        elif item['data_source'] == 'SEED-Bench v2':
            if type(item['data_id']) is list:
                img_path = [os.path.join(self.img_base_path, path) for path in item['data_id']]
            else:
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
            "gt_choices": gt_choices,
            'source': item['data_source'],
            'type': item['data_type']
        }
        if self.ppl_cfg:
            res_dict['options'] = gt_choices
        return res_dict


if __name__ == '__main__':
    seedbench2data = SEEDBench2Dataset(base_data_path='../../../data/SEEDBench2',ppl_cfg=True)
    import ipdb;ipdb.set_trace()