import os
import json
from torch.utils.data import Dataset

class MMEDataset(Dataset):
    task_name = 'VQA'
    dataset_name = 'MME'
    data_type = ['commonsense_reasoning', 
                 'text_translation', 
                 'OCR', 
                 'code_reasoning', 
                 'numerical_calculation', 
                 'color', 
                 'posters', 
                 'count', 
                 'celebrity', 
                 'scene', 
                 'position',
                 'existence', 
                 'landmark',  
                 'artwork']
    def __init__(self,
                 base_data_path,
                 ppl_cfg = True,
                 generative = False,
                 **kwargs
        ):
        self.base_data_path = base_data_path
        self.load_raw_data()
        self.ppl_cfg = ppl_cfg
        self.generative = generative
    
    def load_raw_data(self):
        data = []
        ext_type = ['.jpg', '.png']
        for dtype in self.data_type:
            dir_path = os.path.join(self.base_data_path, dtype)
            if os.path.exists(os.path.join(dir_path, 'questions_answers_YN')):
                annot_dir_path = os.path.join(dir_path, 'questions_answers_YN')
                img_dir_path = os.path.join(dir_path, 'images')
                annot_data_list = os.listdir(annot_dir_path)
                sample_img_name = annot_data_list[0].split('.')[0]
                ext = ''
                for extt in ext_type:
                    if os.path.exists(os.path.join(img_dir_path, sample_img_name + extt)):
                        ext = extt
                        break
                for annot_data_path in annot_data_list:
                    img_path = os.path.join(img_dir_path, annot_data_path.split('.')[0] + ext)
                    if not os.path.exists(img_path):
                        continue
                    with open(os.path.join(annot_dir_path, annot_data_path)) as f:
                        annot_lines = f.readlines()
                        for annot_line in annot_lines:
                            [question, answer] = annot_line.strip().split('\t')
                            data.append(dict(
                                img_path = img_path,
                                question = question,
                                answer = answer,
                                task_type = dtype,
                            ))
            else:
                data_file_list = os.listdir(dir_path)
                ext = ''
                for data_file in data_file_list:
                    for extt in ext_type:
                        if data_file.endswith(extt):
                            ext = extt
                    if ext != '':
                        break
                for data_file in data_file_list:
                    if data_file.endswith('.txt'):
                        img_path = os.path.join(dir_path, data_file.split('.')[0] + ext)
                        if not os.path.exists(img_path):
                            continue
                        with open(os.path.join(dir_path, data_file)) as f:
                            annot_lines = f.readlines()
                            for annot_line in annot_lines:
                                [question, answer] = annot_line.strip().split('\t')
                                data.append(dict(
                                    img_path = img_path,
                                    question = question,
                                    answer = answer,
                                    task_type = dtype,
                                ))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        if self.generative:
            question = question.replace(' Please answer yes or no.','')
        img_path = item['img_path']
        gt_answers = item['answer']
        id = str(item['question_id']) if 'question_id' in item else str(idx)
        res_dict = {
            'id': id,
            "image_path": img_path,
            "question": question,
            "gt_answers": gt_answers,
            'task_type': item['task_type'],
        }
        if self.ppl_cfg:
            res_dict['options'] = ['Yes', 'No']
        return res_dict
