from tqdm import tqdm
import json
from .utils import Base_Metric
import re
import copy

class Winoground_Metric(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
    
    def metric_func(self, answers):
        
        tasks = {}
        answers = sorted(answers, key=lambda x: x['id']) 
        for ans in answers:
            if ans['main_id'] not in tasks:
                tasks[ans['main_id']] = [ans]
            else:
                tasks[ans['main_id']].append(ans)
        
        text_score, image_score, group_score = 0, 0, 0
        for id, task in tasks.items():
            score = []
            for i in task:
                score.append(i['ppl_results'][0])
            tsc, isc =0, 0

            shuffle = i['shuffle']

            if shuffle&1>0:
                tmp = score[0]
                score[0] = score[2]
                score[2] = tmp
                tmp = score[1]
                score[1] = score[3]
                score[3] = tmp
            if shuffle&2>0:
                tmp = score[0]
                score[0] = score[1]
                score[1] = tmp
                tmp = score[2]
                score[2] = score[3]
                score[3] = tmp

            if score[0] < score[2] and score[3] < score[1]:
                tsc = 1
            if score[0] < score[1] and score[3] < score[2]:
                isc = 1
            text_score += tsc
            image_score += isc
            group_score += (tsc+isc)//2

        results={
            'Text': text_score/len(tasks),
            'Image': image_score/len(tasks),
            'Group': group_score/len(tasks),
        }
        #print(f'results for Winoground:\n{results}')

        return results

    def metric(self, answer_path):
        with open(answer_path, 'rb') as f:
            answers = json.load(f)
        results = self.metric_func(answers) 
        print(f'{self.dataset_name}:')
        for key, value in results.items():
            print(f'{key}: {value}')
        return results

class Winoground_Cap_Metric(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
    
    def metric_func(self, answers):
        
        
        #answers = sorted(answers, key=lambda x: x['id']) 
        
        text_score, image_score, group_score = 0, 0, 0
        for task in answers:
            score = task['ppl_results']
            tsc, isc =0, 0
            if score[0] < score[2] and score[3] < score[1]:
                tsc = 1
            if score[0] < score[1] and score[3] < score[2]:
                isc = 1
            text_score += tsc
            image_score += isc
            group_score += (tsc+isc)//2

        results={
            'Text': text_score/len(answers),
            'Image': image_score/len(answers),
            'Group': group_score/len(answers),
        }
        #print(f'results for Winoground:\n{results}')

        return results

    def metric(self, answer_path):
        with open(answer_path, 'rb') as f:
            answers = json.load(f)
        results = self.metric_func(answers) 
        print(f'{self.dataset_name}:')
        for key, value in results.items():
            print(f'{key}: {value}')
        return results