import argparse
import os
import json
import numpy as np
from datasets.utils import *
from tqdm import tqdm
from datasets import load_3Deval_dataset


def detection3d_eval(dataset, pred_data, thres=0.5):
    score = 0
    cnt = 0
    for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
        gt_objects = gt['object']
        text = pred['text']
        bboxes = parse_bbox_3d(text)
        cnt += len(gt_objects)
        for object_info in gt_objects:
            if not classification_acc(object_info['label'], text):
                continue
            for bbox in bboxes:
                iou = cal_iou_3d(object_info['bbox'], bbox)
                if iou > thres:
                    score += 1
                    break
    print(score / cnt)

def grounding3d_eval(dataset, pred_data, thres=0.5):
    score = 0
    cnt = 0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        text = pred['text']
        bboxes = parse_bbox_3d(text)
        cnt += 1
        if len(bboxes) < 1:
            continue
        bbox = bboxes[0]
        iou = cal_iou_3d(gt['object'], bbox)
        if iou > thres:
            score += 1
    print("Acc over {}: {}".format(thres, score / cnt))

def grounding3d(dataset, pred_data):
    grounding3d_eval(dataset, pred_data, thres=0.25)
    grounding3d_eval(dataset, pred_data, thres=0.5)

CHOICE = ['A', 'B', 'C', 'D', 'E', 'F']         # 6 choices in total

def VQAvisionacc(dataset,pred_data):
    import re
    pattern_1 = re.compile(r'The answer is \(?[A-F]\)?\W|the answer is \(?[A-F]\)?\W')
    pattern_2 = re.compile(r'ANSWER: [A-F]')
    pattern_3 = re.compile(r'\([A-F]\)')
    def check_text(text, choices, gt_id):
        text = text.lower()
        if choices[gt_id].lower() not in text:
            return False
        for id, choice in enumerate(choices):
            if id == gt_id:
                continue
            if choice.lower() in text:
                return False
        return True
    def check_option(res_list, gt_char):
        for res in res_list:
            if gt_char not in res:
                return False
        return True
    def check_pattern2(res_list, gt_char):
        pred = res_list[0][-1]
        if pred == gt_char:
            return True
        return False
    score = 0.0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        tmp_score = 0
        gt_choice = gt['gt_choice']
        gt_char = CHOICE[gt_choice]
        pred_text = pred['text']
        pred_text = pred_text
        res_1 = pattern_1.findall(pred_text)
        res_2 = pattern_2.findall(pred_text)
        res_3 = pattern_3.findall(pred_text)
        if len(res_1) != 0:
            if check_option(res_1, gt_char):
                tmp_score = 1.0
        elif len(res_2) != 0:
            if check_pattern2(res_2, gt_char):
                tmp_score = 1.0
        elif len(res_3) != 0:
            if check_option(res_3, gt_char):
                tmp_score = 1.0
        elif check_text(pred_text, gt['gt_choices'], gt_choice):
            tmp_score = 1.0
        score += tmp_score
    print('vision: {}'.format(score / len(dataset)))
    

dataset2evalfunc = {
    'ScanNet': detection3d_eval,
    'ScanRefer': grounding3d,
    'ScanQA_multiplechoice': VQAvisionacc,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument('--answer-file', required=True)
    parser.add_argument('--base-data-path', required=True)
    args = parser.parse_args()
   
    dataset_name = args.dataset_name 
    dataset = load_3Deval_dataset(
        args.base_data_path,
        dataset_name,
        'common',
        batch_size = 1
    ).dataset

    task_name = dataset.task_name
    eval_func = dataset2evalfunc[dataset_name]

    if args.answer_file.endswith('.jsonl'):
        import jsonlines
        pred_data = []
        with open(args.answer_file, 'rb') as f: 
            for item in jsonlines.Reader(f):
                pred_data.append(item)
    elif args.answer_file.endswith('.json'):
        pred_data = json.load(open(args.answer_file,'rb'))
    else:
        file_ext = '.json'
        file_name = task_name + '_' + dataset_name + file_ext
        args.answer_file = os.path.join(args.answer_file, file_name)
        pred_data = json.load(open(args.answer_file, 'rb'))
    print(f'Eval [{args.answer_file}] on {dataset_name}')
    eval_func(dataset, pred_data)
