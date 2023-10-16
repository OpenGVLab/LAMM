from tqdm import tqdm
from .utils import Base_Metric

class Caption(Base_Metric):
    def __init__(self, dataset_name, strategy = 'direct', **kwargs):
        super().__init__(dataset_name)
        
        self.strategy = strategy
        assert self.strategy in ['direct', 'acc', 'top_acc']

    def normal_metric_func(self, answers):
        from nltk.translate.bleu_score import sentence_bleu
        from .utils import parse_caption_sentence
        from rouge import Rouge
        self.rouge = Rouge(metrics=["rouge-l"])
        self.bleu = sentence_bleu
        self.parse_sentence = parse_caption_sentence
        bleu_score, rouge_score = 0, 0
        for item in tqdm(answers, desc="Running Metric"):
            gt = item['gt_answers']
            pred = item['answer']
            if pred == '':
                continue
            pred_caption = self.parse_sentence(pred)
            pred_captions = pred_caption.split('.')
            while '' in pred_captions:
                pred_captions.remove('')
            references = [sentence.replace('.','').split() for sentence in gt]
            tmp_bleu_score = self.bleu(references, pred.split(),  (1./4., 1./4., 1./4., 1./4.))
            tmp_bleu_score = max(tmp_bleu_score, self.bleu(references, pred_caption.split(), (1./4., 1./4., 1./4., 1./4.)))
            for caption in pred_captions:
                tmp_bleu_score = max(tmp_bleu_score, self.bleu(references, caption.split(), (1./4., 1./4., 1./4., 1./4.)))
            
            tmp_rouge_score = 0
            for sentence in gt:
                tmp_rouge_score = max(tmp_rouge_score, self.rouge.get_scores(pred, sentence, avg=True)['rouge-l']['f'])
                tmp_rouge_score = max(tmp_rouge_score, self.rouge.get_scores(pred_caption, sentence, avg=True)['rouge-l']['f'])
                for caption in pred_captions:
                    tmp_rouge_score = max(tmp_rouge_score, self.rouge.get_scores(caption, sentence, avg=True)['rouge-l']['f'])
            bleu_score += tmp_bleu_score
            rouge_score += tmp_rouge_score
        bleu_score = bleu_score * 100 / len(answers)
        rouge_score = rouge_score * 100 / len(answers)
        return dict(
            BLEU4 = bleu_score,
            Rouge = rouge_score,
        )
    
    def acc_metric_func(self, answers):
        score = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt = item['gt_answers']
            pred = item['answer']
            if pred in gt:
                score += 1
        score = score * 100 / len(answers)
        return dict(
            ACC = score
        )

    def top_acc_metric_func(self, answers):
        # top_acc calculates the percentage of the correct caption in the top50% of the results
        import numpy as np
        score = 0
        for item in tqdm(answers, desc="Running Metric"):
            ppl_value = item['ppl_results']
            pred_sort = np.array(ppl_value).argsort()
            ppl_len = len(ppl_value)
            assert ppl_len % 2 == 0
            tmp_score = 0
            for i in range(ppl_len//2):
                if pred_sort[i] < ppl_len//2:
                    tmp_score+=1
            score+= tmp_score/ (ppl_len // 2)
        score = score * 100 / len(answers)
        return dict(
            ACC = score
        )

    def metric_func(self, answers):
        if self.strategy == 'direct':
            return self.normal_metric_func(answers)
        elif self.strategy == 'acc':
            return self.acc_metric_func(answers)
        else:
            return self.top_acc_metric_func(answers)