from tqdm import tqdm

from .utils import Base_Metric, parse_caption_sentence

class SVT_OCR(Base_Metric):

    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)

    def metric_func(self, answers):
        score = 0.0
        for item in tqdm(answers, desc="Running Metric"):
            gt_word_list = item['gt_answers']
            pred_text = item['answer']
            pred_word_list = parse_caption_sentence(pred_text).lower().split()
            correct = 0
            for word in gt_word_list:
                if word.lower() in pred_word_list:
                    correct += 1
            tmp_score = correct / len(gt_word_list)
            score += tmp_score

        return dict(
            ACC = score/len(answers),
        )