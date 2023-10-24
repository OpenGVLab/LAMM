from tqdm import tqdm
from .utils import Base_Metric
import re
import copy

class Answer_Extractor:
    def __init__(self, content_only = False) -> None:
        self.choices = 'ABCDEFG'
        self.content_only = content_only
    # Prefetch Answers
    def infer_option(self, answer):
        def get_unit_option(splits, choices='ABCD', prefix='', suffix=''):
            res = []
            for c in choices:
                if prefix + c + suffix in splits:
                    res.append(c)
            return res
        splits = [x.strip() for x in answer.split()]

        # no prefix match
        no_prefix_option = get_unit_option(splits, self.choices)
        if len(no_prefix_option) == 1:
            if 'A' not in splits or len(splits)<3:
                return no_prefix_option[0]

        # prefix match
        tups = [('(', ')'), ('(', ').'), ('', '.'), ('', ','), ('', ':'), ('', ')'), ('', ').'), 
                (':', ''), (':', ','), (':', '.'), (':', ')'), (':', ').')]
        for tup in tups:
            prefix_option = get_unit_option(splits, self.choices, prefix=tup[0], suffix=tup[1])
            if len(prefix_option) == 1:
                return prefix_option[0]
        return None

    def infer_text(self, answer, choices):
        answer = answer.lower()
        assert isinstance(choices, list)
        gt_choices = {}
        for idx, k in enumerate(choices):
            gt_choices[self.choices[idx]] = str(k).lower()
        cands = []
        for key, value in gt_choices.items():
            if value in answer:
                cands.append(key)
        if len(cands) == 1:
            return cands[0]
        return None

    def preprocess_text(self, answer):
        output_text = answer
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        # output_text = output_text.strip('</s><s>')
        # output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        # mmbench direct pattern
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_text)
        if len(res) > 0:
            return '(' + res[0][:-1] + ')'
        # ppl pattern
        pattern = re.compile(r'\([A-Z]')
        res = pattern.findall(output_text)
        if len(res) > 0:
            return res[0] + ')'
        return output_text

    def fetch_answer(self, answer, choices):
        if not self.content_only:
            answer = self.preprocess_text(answer)
            copt = self.infer_option(answer)
            if copt:
                return copt, 1, 0
        if answer in choices:
            return self.choices[choices.index(answer)], 0, 1
        return self.infer_text(answer, choices), 0, 1

class VQA(Base_Metric):
    CHOICE = 'ABCDEFG'
    def __init__(self, dataset_name, content_only = False, **kwargs):
        super().__init__(dataset_name)
        self.answer_extractor = Answer_Extractor(content_only)
        self.match = 0

    def metric_func(self, answers):
        score = 0.0
        for item in tqdm(answers, desc="Running Metric"):
            gt_choice = item['gt_choice']
            gt_char = self.CHOICE[gt_choice]
            pred_text = item['answer']
            pred_option, _, _ = self.answer_extractor.fetch_answer(pred_text, item['gt_choices'])
            if pred_option:
                self.match +=1
            if pred_option == gt_char:
                score += 1.0
        score = score/len(answers) * 100
        return dict(
            ACC = score, 
            match_ratio = self.match /(len(answers)) * 100
        )

class MMBenchVQA(Base_Metric):
    def __init__(self, dataset_name, content_only = False):
        super().__init__(dataset_name)
        self.choices = 'ABCD'
        self.match_option = 0
        self.match_content = 0
        self.answer_extractor = Answer_Extractor(content_only)

    def eval_sub_data(self, sub_data, answer_map):
        lt = len(sub_data)
        GT, PRED = [], []
        result = 1
        for i in range(lt):
            item = sub_data[i]
            idx = item['id']
            GT.append(self.choices[answer_map[idx]])
            pred_answer, option_match, content_match = self.answer_extractor.fetch_answer(item['answer'], item['gt_choices'])
            PRED.append(pred_answer)
            if pred_answer is not None:
                self.match_content += content_match
                self.match_option += option_match
                if GT[-1] != PRED[-1]:
                    result = 0
            else:
                result = 0
        return result

    def metric_func(self, answers):
        vanilla_score, circular_score = 0.0, 0.0
        vanilla_cnt = 0
        result = {}
        answer_map = {} # gt
        cnt = len(answers)
        for item in answers:
            answer_map[item['id']] = item['gt_choice']
        answers = sorted(answers, key = lambda i: int(i['id']))
        for i in tqdm(range(len(answers)), desc="Running Metric"):
            idx = answers[i]['id']
            main_idx = str(int(idx) % int(1e6))
            if main_idx in result:
                continue
            ## vanilla
            vanilla_cnt += 1
            pred_option, _, _ = self.answer_extractor.fetch_answer(answers[i]['answer'], answers[i]['gt_choices'])
            if pred_option == self.choices[answer_map[answers[i]['id']]]:
                vanilla_score += 1
            
            sub_data = []
            for j in range(len(answers)):
                if int(answers[j]['id']) % int(1e6) == int(idx):
                    sub_data.append(answers[j])
            out = self.eval_sub_data(sub_data, answer_map)
            circular_score += out
            result[main_idx] = out

        return dict(
            vanilla_acc = vanilla_score / vanilla_cnt * 100,
            circular_acc = circular_score / vanilla_cnt * 100,
            option_match = self.match_option / cnt * 100,
            content_match = self.match_content /cnt *100,
        )
    
class MMEVQA(Base_Metric):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        from .utils import Cleaner
        self.cleaner = Cleaner()
        self.cnt_dict = {
            "existence": 0, 
            "count": 0, 
            "position": 0, 
            "color": 0,
            "posters": 0, 
            "celebrity": 0, 
            "scene": 0, 
            "landmark": 0, 
            "artwork": 0, 
            "OCR": 0,
            "commonsense_reasoning": 0,
            "numerical_calculation": 0, 
            "text_translation": 0, 
            "code_reasoning":0
        }
    def parse_pred_ans(self, pred_ans):
        pred_ans = pred_ans.lower()
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            pred_ans = self.cleaner.clean(pred_ans)
            options = ['yes', 'no']
            answers = []
            for option in options:
                if option in pred_ans:
                    answers.append(option)
            if len(answers) != 1:   
                pred_label = 'other'
            else:
                pred_label = answers[0]

        return pred_label

    def metric_func(self, answers):
        cnt_dict = copy.deepcopy(self.cnt_dict)
        acc_dict = copy.deepcopy(self.cnt_dict)
        acc_plus_dict = copy.deepcopy(self.cnt_dict)
        acc_plus_cnt_dict = copy.deepcopy(self.cnt_dict)
        accplus = dict()
        for item in tqdm(answers, desc="Running Metric"):
            gt_label = item['gt_answers'].lower()
            pred_text = item['answer']
            image_path = item['image_path']
            if image_path not in accplus:
                accplus[image_path] = 0
                acc_plus_cnt_dict[item['task_type']] += 1
            pred_label = self.parse_pred_ans(pred_text)
            cnt_dict[item['task_type']] += 1
            if pred_label == gt_label:
                acc_dict[item['task_type']] += 1
                accplus[image_path] += 1
                if accplus[image_path] == 2:
                    acc_plus_dict[item['task_type']] += 1

        results_dict = dict()
        acc_overall = 0
        cnt_overall = 0
        acc_plus_overall = 0
        cnt_plus_overall = 0
        for key in acc_dict.keys():
            results_dict[key] = acc_dict[key] / (cnt_dict[key]+ 1e-6) * 100
            results_dict[f'{key}_plus'] = acc_plus_dict[key] / (acc_plus_cnt_dict[key] + 1e-6) * 100
            acc_overall += acc_dict[key]
            cnt_overall += cnt_dict[key]
            acc_plus_overall += acc_plus_dict[key]
            cnt_plus_overall += acc_plus_cnt_dict[key]
        results_dict.update(overall = acc_overall/cnt_overall * 100, overall_plus = acc_plus_overall / cnt_plus_overall * 100)
        return results_dict
