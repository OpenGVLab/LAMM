from tqdm import tqdm
from .utils import Base_Metric

class Counting(Base_Metric):
    def __init__(self, dataset_name, inference_type = 'direct', **kwargs):
        super().__init__(dataset_name)
        self.inference_type = inference_type
        assert self.inference_type in ['direct', 'ppl']
        from .utils import ennum2numerical
        self.parse_num_func = ennum2numerical
    
    def mae_metric(self, answers):
        score = 0
        for item in tqdm(answers, desc="Running MAE Metric"):
            gt_num = item['gt_answers']
            text = item['answer']
            pred_num = self.parse_num_func(text)
            score += min(gt_num, abs(pred_num-gt_num))
        return score / len(answers)

    def acc_metric(self, answers):
        score = 0
        for item in tqdm(answers, desc="Running ACC Metric"):
            gt_num = item['gt_answers']
            text = item['answer']
            pred_num = self.parse_num_func(text)
            score += (pred_num == gt_num)
        return score / len(answers) * 100

    def metric_func(self, answers):
        res_dict = {}
        if self.inference_type == 'direct':
            res_dict['MAE'] = self.mae_metric(answers)
        res_dict['ACC'] = self.acc_metric(answers)
        return res_dict
