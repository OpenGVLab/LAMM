import argparse
import os
import json
import numpy as np
from datasets.utils import *
from tqdm import tqdm
from datasets import load_2Deval_dataset


def inside_bbox_eval(dataset, pred_data):
    correct_cnt, total_cnt = 0,0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        gt_objects = gt['object']
        text = pred['text']
        keypoints = parse_keypoints(text)
        if len(keypoints) == 0:
            total_cnt += len(gt_objects)
            continue
        for object in gt_objects:
            total_cnt += 1
            if not classification_acc(object['label'], text):
                continue
            gt_bbox = object['bbox']
            if check_inside_bbox(keypoints, gt_bbox):
                correct_cnt += 1
    return correct_cnt, total_cnt


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


def point_distance_eval(dataset, pred_data):
    correct_cnt, total_cnt = 0,0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        points = gt['points']
        width, height = gt['image'].size
        points = np.array(points) / np.array([width, height])
        if len(points)>=10:
            continue
        pred_text = pred['text']
        keypoints = parse_keypoints(pred_text)
        total_cnt += len(points)
        for point in points:
            if correct_keypoints(keypoints, point):
                correct_cnt += 1
    return correct_cnt, total_cnt


dataset2evalfunc = {
    'VOC2012': inside_bbox_eval,
    'LSP':inside_human_bbox_eval,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer-dir', required=True)
    parser.add_argument('--base-data-path', required=True)
    args = parser.parse_args()
    answer_base_dir = args.answer_dir
    dataset_list = ['VOC2012', 'LSP']
    total_correct, total_num = 0,0
    for dataset_name in dataset_list:
        load_data = dataset_name in ['VOC2012', 'LSP']
        dataset = load_2Deval_dataset(
            args.base_data_path,
            dataset_name,
            'locating',
            load_data=load_data,
            batch_size=1,
        ).dataset
        task_name = dataset.task_name
        answer_file = os.path.join(answer_base_dir,task_name+'_'+dataset_name+'.json')
        pred_data = json.load(open(answer_file,'rb'))
        eval_func = dataset2evalfunc[dataset_name]
        correct_cnt, total_cnt = eval_func(dataset, pred_data)
        total_correct += correct_cnt
        total_num += total_cnt
    print(f'Total num:{total_num} Correct num:{total_correct} ACC:{total_correct/total_num}')

