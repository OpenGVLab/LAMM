import json
from datasets import load_2Deval_dataset
import os
import random
from tqdm import tqdm
import openai
import time
import argparse

openai.api_key=''

GT_Template = """
[Question]\n
{}\n\n
[Ground Truth answer]\n
{}\n\n
"""
Ans_Template = """
[The Start of Assistant {}'s Answer]\n
{}\n
[The End of Assistant {}'s Answer]\n
"""

SYS = """
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. \n
Please provide a ranklist due to the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives a ranking. \n
Please first output a single line containing only two values indicating the ranking for Assistant A, B respectively. The two ranking values are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
"""

system_msg = {
    'CIFAR10': 'image classification',
    'SVT': 'optical character organization\(OCR\)',
    'flickr30k':'image caption',
    'CelebA(Hair)': 'facial classification',
    'CelebA(Smile)': 'facial classification',
    'UCMerced':' fine-grained classification',
    'VOC2012':'object detection',
    'AI2D':'vision question answering\(VQA\)',
    'SQAimage':'vision question answering\(VQA\)',
}

eval_data_len = {
    'CIFAR10': 150,
    'SVT': 50,
    'flickr30k': 200,
    'CelebA(Hair)': 50,
    'CelebA(Smile)': 50,
    'UCMerced': 50,
    'VOC2012': 50,
    'AI2D': 150,
    'SQAimage': 250,
}

def option_text(id, choices):
    option = ['A','B','C','D','E','F','G']
    return '(' + option[id] + ') ' + choices[id] 

def get_ground_truth_data(gt_data):
    if 'label' in gt_data:
        return gt_data['label']
    if 'object' in gt_data:
        return gt_data['object']
    if 'word_list' in gt_data:
        return str(gt_data['word_list']).lower()
    if 'sentences' in gt_data:
        return gt_data['sentences']
    if 'attr' in gt_data:
        if gt_data['attr'] == '1':
            return 'Yes'
        elif gt_data['attr'] == '-1':
            return 'No'
        else:
            return gt_data['attr']
    if 'gt_choice' in gt_data:
        choices = gt_data['gt_choices']
        gt_choice = gt_data['gt_choice']
        return option_text(gt_choice,choices)


def generate_query(gt_data, pred_data_list):
    ID_Name = ['A', 'B', 'C', 'D']
    pred_num = len(pred_data_list)
    query = ''
    ground_truth_data = get_ground_truth_data(gt_data) 
    query += GT_Template.format(gt_data['query'], ground_truth_data)
    for i in range(pred_num):
        query+=Ans_Template.format(ID_Name[i], pred_data_list[i]['text'], ID_Name[i])
    query += SYS
    return query

def GPT_Metric(base_data_path,
               answer_dir,
               compare_dir,
               response_dir):
    answer_base_dir_list = [answer_dir, compare_dir]
    query_list = []
    for dataset_name, system in system_msg.items():
        gt_dataset = load_2Deval_dataset(base_data_path,
                                        dataset_name,
                                        'common',
                                        load_data = False).dataset
        task_name = gt_dataset.task_name
        answer_file_name = task_name + '_' +dataset_name + '.json'
        gt_data_len = len(gt_dataset)

        ids = [i for i in range(gt_data_len)]
        random.shuffle(ids)
        ids = ids[:eval_data_len[dataset_name]]

        answer_list = []
        
        for answer_base_dir in answer_base_dir_list:
            answer_list.append(json.load(open(os.path.join(answer_base_dir, answer_file_name),'rb')))
        
        print(f'loading {dataset_name}')
        for id in tqdm(ids):
            pred_data_list = []
            try:
                for j in range(len(answer_list)): 
                    pred_data_list.append(answer_list[j][id])
            except:
                continue
            query_list.append(dict(
                text = generate_query(gt_dataset[id], pred_data_list),
                system = system
            ))
    response_path = os.path.join(response_dir, 'gpt_metric.jsonl')
    ans_file = open(response_path,'w')
    res_list = []
    print('GPT evaluating ...')
    for query in tqdm(query_list):
        messages = [{"role":"system", "content": "You are a helpful and precise assistant for checking the quality of the answer from two different multimodal large language models. They are performing {}. ".format(query['system'])}]
        messages.append({"role":"user", "content":'\n'.join(query['text'])})
        try:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        except:
            continue
        res_list.append(response)
        ans_file.write(json.dumps(response) + "\n")
        ans_file.flush()
        time.sleep(2)
    return res_list
    
def parse_score(text):
    text = text.split('\n')[0]
    rank_list = text.split()
    if rank_list[0] == '1' and rank_list[1] == '2':
        return 1
    elif rank_list[0] == '2' and rank_list[1] == '1':
        return 2
    return None

def eval_score(response_list):
    scorea, scoreb, cnt = 0, 0, 0
    for item in response_list:
        text = item['choices'][0]['message']['content']
        if parse_score(text) == 1:
            scorea += 100
            scoreb += 0
            cnt+=1
        elif parse_score(text) == 2:
            scorea += 0
            scoreb += 100
            cnt+=1
        else:
            pass
    print(scorea/cnt, scoreb/cnt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-data-path', required=True)
    parser.add_argument('--answer-dir', required=True)
    parser.add_argument('--compare-dir', required=True)
    parser.add_argument('--response-dir', required=True)
    args = parser.parse_args()
    response_list = GPT_Metric(args.base_data_path,
                               args.answer_dir,
                               args.compare_dir,
                               args.response_dir)
    eval_score(response_list)