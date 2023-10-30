from tqdm import tqdm
import numpy as np
from .utils import Base_Metric, parse_keypoints, classification_acc, check_inside_bbox
from ..models.utils import get_image

def inside_human_bbox_eval(dataset, pred_data):
    correct_cnt, total_cnt = 0,0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        total_cnt += 1
        joints = gt['gt_joints']
        text = pred['text']
        keypoints = parse_keypoints(text)
        if len(keypoints) == 0:
            continue
        joints = np.array(joints)[:,:2]
        gt_bbox = np.concatenate([joints.min(axis=0), joints.max(axis=0)])
        width, height = gt['image'].size
        for keypoint in keypoints:
            keypoint[0]*=width
            keypoint[1]*=height
        if check_inside_bbox(keypoints, gt_bbox):
            correct_cnt += 1
    return correct_cnt, total_cnt

class InsideBbox(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
    
    def metric_func(self, answers):
        correct_cnt, total_cnt = 0,0
        for item in tqdm(answers, desc="Running Metric"):
            gt = item['gt_answers']
            pred = item['answer']
            keypoints = parse_keypoints(pred)
            if len(keypoints) == 0:
                total_cnt += len(gt)
                continue
            for object in gt:
                total_cnt += 1
                if not classification_acc(object['label'], pred):
                    continue
                gt_bbox = object['bbox']
                if check_inside_bbox(keypoints, gt_bbox):
                    correct_cnt += 1
        return dict(
            ACC = (correct_cnt / total_cnt) * 100
        )

class InsideHumanBbox(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
    
    def metric_func(self, answers):
        correct_cnt, total_cnt = 0,0
        for item in tqdm(answers, desc="Running Metric"):
            total_cnt += 1
            joints = item['gt_answers']
            text = item['answer']
            keypoints = parse_keypoints(text)
            if len(keypoints) == 0:
                continue
            joints = np.array(joints)[:,:2]
            gt_bbox = np.concatenate([joints.min(axis=0), joints.max(axis=0)])
            img = get_image(item['image_path'])
            width, height = img.size
            for keypoint in keypoints:
                keypoint[0]*=width
                keypoint[1]*=height
            if check_inside_bbox(keypoints, gt_bbox):
                correct_cnt += 1
        return dict(
            ACC = (correct_cnt / total_cnt) * 100
        )