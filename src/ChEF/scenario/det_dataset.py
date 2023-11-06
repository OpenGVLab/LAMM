import os
import json
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import numpy as np
import copy


def cal_iou(bbox1, bbox2):
    ixmin = np.maximum(bbox1[0], bbox2[0])
    iymin = np.maximum(bbox1[1], bbox2[1])
    ixmax = np.minimum(bbox1[2], bbox2[2])
    iymax = np.minimum(bbox1[3], bbox2[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw * ih

    uni = ((bbox2[2] - bbox2[0] ) * (bbox2[3] - bbox2[1]) +
            (bbox1[2] - bbox1[0]) *
            (bbox1[3] - bbox1[1]) - inters)
    
    overlaps = inters / uni
    return overlaps

def generate_scaled_translated_bbox(bbox):
    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    scale = random.uniform(0.5, 1.5)

    new_width = width * scale
    new_height = height * scale

    new_x1 = x1 + (width - new_width) / 2
    new_y1 = y1 + (height - new_height) / 2
    new_x2 = new_x1 + new_width
    new_y2 = new_y1 + new_height

    translate_x = random.uniform(-width/2, width/2)
    translate_y = random.uniform(-height/2, height/2)

    new_x1 = max(0, new_x1 + translate_x)
    new_y1 = max(0, new_y1 + translate_y)
    new_x2 = min(1, new_x2 + translate_x)
    new_y2 = min(1, new_y2 + translate_y)


    return [new_x1, new_y1, new_x2, new_y2]

def check_bbox(bbox, gt_bboxes):
    bbox = np.array(bbox)
    if not np.all(bbox>=0.0) or not np.all(bbox<=1.0):
        return False
    for gt_bbox in gt_bboxes:
        if cal_iou(np.array(gt_bbox), bbox) > 0.5:
            return False
    return True

def generate_random_bbox():
    centerx = random.uniform(0,1)
    centery = random.uniform(0,1)
    width = random.uniform(0,1)
    height = random.uniform(0,1)
    new_x1 = max(0, centerx - width/2)
    new_y1 = max(0, centery - height/2)
    new_x2 = min(1, centerx + width/2)
    new_y2 = min(1, centery + height/2)
    return [new_x1, new_y1, new_x2, new_y2]

class VOC2012Dataset(Dataset):
    task_name = 'detection'
    dataset_name = 'VOC2012'
    def __init__(self, 
                 base_data_path = 'data/LAMM/LAMM/LAMM-Dataset/2D_Benchmark/',
                 ppl_cfg = None,
                 option_template = 'default',
                 **kwargs):
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,'meta_file', 'Detection_VOC2012.json')
        self.data = json.load(open(json_path,'rb'))
        self.ppl_cfg = ppl_cfg
        assert option_template in ['default', 'shikra', 'kosmos']
        self.option_template = option_template
        if self.ppl_cfg:
            self.negative_opt_num = self.ppl_cfg.get('negative_opt_num', 3)
            self.random_seed = self.ppl_cfg.get('random_seed', 0)
            random.seed(self.random_seed)
            self.load_ppl_options()
    
    def generate_negative_bbox(self, gt_bboxes, other_class_bboxes):
        bboxes = gt_bboxes + other_class_bboxes
        candidate_list = []
        for bbox in bboxes:
            for _ in range(self.negative_opt_num):
                random_scaled_bbox = generate_scaled_translated_bbox(bbox)
                if check_bbox(random_scaled_bbox, gt_bboxes):
                    candidate_list.append(random_scaled_bbox)
        return candidate_list

    def generate_random_bbox(self, gt_bboxes, num):
        if num <= 0 :
            return []
        candidates = []
        while len(candidates) < num:
            random_bbox = generate_random_bbox()
            if check_bbox(random_bbox, gt_bboxes):
                candidates.append(random_bbox)
        return candidates
    
    def load_ppl_options(self):
        print('----generate ppl negative options----')
        self.all_class_names = []
        self.new_data_list = []
        for i in tqdm(range(len(self.data)), desc="Running class set"):
            objects = self.data[i]['object']
            data_item = {}
            for object in objects:
                if object['label'] not in self.all_class_names:
                    self.all_class_names.append(object['label'])
                if object['label'] not in data_item:
                    data_item[object['label']] = []
                data_item[object['label']].append(object['bbox'])
            self.new_data_list.append(data_item)

    def __len__(self):
        return len(self.data)
    
    def bbox2pploption(self, bbox, class_name = None):
        def point2index(x, y):
            x = int(x * 32)
            y = int(y * 32)
            index = y * 32 + x
            return str(index).zfill(4)
        if self.option_template == 'shikra':
            return  f'{class_name}[{bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f}]'
        elif self.option_template == 'default':
            return f'[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]'
        elif self.option_template == 'kosmos':
            return f'<object><patch_index_{point2index(bbox[0],bbox[1])}><patch_index_{point2index(bbox[2], bbox[3])}></object>'

    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['id']) if 'id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': os.path.join(self.base_data_path,self.data[index]['image']),
            'gt_answers': self.data[index]['object']
        }
        if self.ppl_cfg:
            data_item = self.new_data_list[index]
            gt_answers = []
            classification_options = []
            gt_classes = [key for key in data_item.keys()]
            tmp = copy.deepcopy(self.all_class_names)
            for gt_class in gt_classes:
                tmp.remove(gt_class)
            for gt_class in gt_classes:
                random.shuffle(tmp)
                classification_options.append([gt_class] + tmp[:self.negative_opt_num])

            grounding_options = []
            for key in data_item.keys():
                gt_bboxes = data_item[key]
                tmp = copy.deepcopy(data_item)
                del tmp[key]
                other_class_bboxes = [bbox for value in data_item.values() for bbox in value]
                candidates = self.generate_negative_bbox(gt_bboxes, other_class_bboxes)
                random_candidates = self.generate_random_bbox(gt_bboxes, self.negative_opt_num - len(candidates))
                candidates += random_candidates
                assert len(candidates) >= self.negative_opt_num
                candidates = [self.bbox2pploption(bbox, key) for bbox in candidates]
                for gt_bbox in gt_bboxes:
                    gt_answers.append(dict(
                        label = key,
                        bbox = gt_bbox
                    ))
                    random.shuffle(candidates)
                    grounding_options.append(dict(
                        fore_label = key,
                        options = [self.bbox2pploption(gt_bbox, key)] + candidates[:self.negative_opt_num]
                    ))
            res_dict['gt_answers'] = gt_answers
            res_dict['classification_options'] = classification_options
            res_dict['grounding_options'] = grounding_options
            
        return res_dict
