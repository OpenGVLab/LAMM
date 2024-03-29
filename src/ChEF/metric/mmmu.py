from tqdm import tqdm
from .utils import Base_Metric
from typing import Dict
import numpy as np
from .mmmu_utils import parse_multi_choice_response, parse_open_response, eval_multi_choice, eval_open

CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

"""Response Parsing and Evaluation for various models"""

OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
class MMMU_Metric(Base_Metric):

    def __init__(self, dataset_name, ppl=False, **kwargs):
        super().__init__(dataset_name)
        self.ppl = ppl

    def metric_func(self, answers):
        res_dict = {}
        for value in CAT_SHORT2LONG.values():
            res_dict[value] = {'acc': 0, 'correct_nums': 0, 'sample_nums': 0}

        for item in tqdm(answers, desc="Running Metric"):
            cate = item['subject']
            answer = item['answer']
            
            if item['question_type'] == 'multiple-choice': 
                all_choices = [OPTIONS[i] for i in range(len(item['options']))]
                index2ans = {OPTIONS[i]:item['options'][i] for i in range(len(item['options']))}
                pred_i = parse_multi_choice_response(answer, all_choices, index2ans)
            else:
                pred_i = parse_open_response(answer)
            gold_i = item['gt_choice']

            if item['question_type'] == 'multiple-choice':
                correct = eval_multi_choice(gold_i, pred_i)
            else: # open question
                correct = eval_open(gold_i, pred_i)
            item['metric_result'] = correct
            res_dict[cate]['sample_nums'] += 1
            item['parsed_answer'] = answer
            if correct:
                res_dict[cate]['correct_nums'] += 1
        correct_nums, total = 0, 0
        for key in res_dict.keys():
            n = res_dict[key]['sample_nums'] 
            correct_nums += res_dict[key]['correct_nums']
            total += n
            if n != 0:
                res_dict[key]['acc'] = res_dict[key]['correct_nums']  / n * 100
        res_dict['Overall'] = {
            'acc': correct_nums / total * 100,
            'sample_nums':  total
        }
        return res_dict, answers
