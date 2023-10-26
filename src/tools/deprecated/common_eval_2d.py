import argparse
import os
import json
import numpy as np
from datasets.utils import *
from tqdm import tqdm
from datasets import load_2Deval_dataset


def detection_eval(dataset, pred_data, thres=0.5):
    score = 0
    cnt = 0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        gt_objects = gt['object']
        text = pred['text']
        bboxes = parse_bbox(text)
        cnt += len(gt_objects)
        for object_info in gt_objects:
            if not classification_acc(object_info['label'], text):
                continue
            for bbox in bboxes:
                iou = cal_iou(object_info['bbox'], bbox)
                if iou > thres:
                    score += 1
                    break
    print(score / cnt)


def SVT_eval(dataset, pred_data):
    score = 0.0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        gt_word_list = gt['word_list']
        pred_text = pred['text']
        pred_word_list = parse_sentence(pred_text).lower().split()
        correct = 0
        for word in gt_word_list:
            if word.lower() in pred_word_list:
                correct += 1
        tmp_score = correct / len(gt_word_list)
        score += tmp_score
    print(score / len(dataset))


def FSC_eval(dataset, pred_data):
    score, ten_score, ten_cnt = 0.0, 0.0, 0.0
    for gt, pred in zip(dataset, pred_data):
        gt_label = gt['num']
        pred_text = pred['text']
        pred_text = pred_text.lower()
        num = ennum2numerical(pred_text)
        if num is None:
            num = 0
        MSE = abs(gt_label - num)
        score += MSE
        if gt_label < 10:
            ten_score += MSE
            ten_cnt += 1
    print('total:{}, less than 10:{} '.format(score / len(dataset), ten_score / ten_cnt))


CHOICE = ['A', 'B', 'C', 'D', 'E']


def VQAvisionacc(dataset, pred_data):
    import re
    pattern_1 = re.compile(r'The answer is \(?[A-E]\)?\W|the answer is \(?[A-E]\)?\W')
    pattern_2 = re.compile(r'ANSWER: [A-E]')
    pattern_3 = re.compile(r'\([A-E]\)')
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
        if len(res_1) !=0 :
            if check_option(res_1, gt_char):
                tmp_score = 1.0
        elif len(res_2) !=0:
            if check_pattern2(res_2, gt_char):
                tmp_score = 1.0
        elif len(res_3) != 0:
            if check_option(res_3, gt_char):
                tmp_score = 1.0
        elif check_text(pred_text, gt['gt_choices'], gt_choice):
            tmp_score = 1.0
        score += tmp_score
    print('vision: {}'.format(score / len(dataset)))
    

def BLEU4(dataset, pred_data):
    from nltk.translate.bleu_score import sentence_bleu
    score = 0.0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        gt_sentences = gt['sentences']
        pred_text = pred['text']
        pred_caption = parse_sentence(pred_text)
        references = []
        for sentence in gt_sentences:
            references.append(sentence.replace('.','').split())
        pred_captions = pred_caption.split('.')
        tmp_score = 0.0
        for caption in pred_captions:
            tmp_score = max(tmp_score, sentence_bleu(references, caption.split(), (1./4., 1./4., 1./4., 1./4.)) )
        score += tmp_score
    print(score * 100 / len(dataset))


def UCMerced_eval(dataset, pred_data):
    score = 0.0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        gt_label = gt['label']
        pred_text = pred['text']
        if gt_label in pred_text:
            score += 1
    print(score / len(dataset))


def CelebA_smile_eval(dataset, pred_data):
    score = 0.0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        gt_label = gt['attr']
        pred_text = pred['text']
        text = pred_text.lower()
        words = word_tokenize(text)
        words = [word for word in words if word not in string.punctuation]
        if 'no' in words or 'not' in words:
            pred_label = '-1'
        else:
            pred_label = '1'
        if pred_label == gt_label:
            score += 1.0
    print(score / len(dataset))


def CelebA_hair_eval(dataset, pred_data):
    score = 0.0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        gt_label = gt['attr']
        pred_text = pred['text']
        if classification_acc(gt_label, pred_text):
            score += 1.0
    print(score / len(dataset))


def CIFAR10(dataset, pred_data):
    score = 0.0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        gt_label = gt['label']
        pred_text = pred['text']
        if classification_acc(gt_label, pred_text):
            score += 1.0
    print(score / len(pred_data)) 


def PCK(dataset, pred_data):
    score = 0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        pred_text = pred['text']
        joints = gt['gt_joints']
        width, height = gt['image'].size
        joints = np.array(joints)
        joints[:, :2] = joints[:, :2] / np.array([width, height])
        tmp_score = 0.0
        for id, (_, value) in enumerate(pred_text.items()):
            keypoints = parse_keypoints(value)
            gt_joint = joints[id]
            if correct_keypoints(keypoints, gt_joint[:2]):
                tmp_score + 1.0
        score += tmp_score / 14
    print(score / len(pred_data))


dataset2evalfunc = {
    'VOC2012': detection_eval,
    'SQAimage': VQAvisionacc,
    'SVT': SVT_eval,
    'flickr30k': BLEU4,
    'FSC147' :FSC_eval,
    'UCMerced': UCMerced_eval,
    'CelebA(Smile)':CelebA_smile_eval,
    'CelebA(Hair)':CelebA_hair_eval,
    'CIFAR10': CIFAR10,
    'AI2D': VQAvisionacc,
    'LSP':PCK,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument('--answer-file', required=True)
    parser.add_argument('--base-data-path', required=True)
    args = parser.parse_args()
   
    dataset_name = args.dataset_name 
    load_data = dataset_name in ['FSC147', 'LSP']
    dataset = load_2Deval_dataset(
        args.base_data_path,
        dataset_name,
        'common',
        load_data = load_data,
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
