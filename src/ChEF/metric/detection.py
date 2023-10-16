from tqdm import tqdm
from .utils import Base_Metric
class Detection(Base_Metric):
    def __init__(self, dataset_name, threshold = 0.5, inference_type = 'direct', **kwargs):
        super().__init__(dataset_name)
        from .utils import classification_acc, parse_bbox, cal_iou
        self.check_func = classification_acc
        self.parse_func = parse_bbox
        self.iou = cal_iou
        self.threshold = threshold
        self.inference_type = inference_type
        assert self.inference_type in ['direct', 'ppl']

    def direct_metric(self, answers):
        score = 0
        rec_cnt = 0
        pre_cnt = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt_objects = item['gt_answers']
            text = item['answer']
            if isinstance(text, str):
                bboxes = self.parse_func(text)
                cnt += len(gt_objects)
                for object_info in gt_objects:
                    if not self.check_func(object_info['label'], text):
                        continue
                    for bbox in bboxes:
                        iou = self.iou(object_info['bbox'], bbox)
                        if iou > self.threshold:
                            score += 1
                            break
            elif isinstance(text,dict):
                pred_bboxes = {key: self.parse_func(value) if value is not None else [] for key, value in text.items()}
                pre_cnt += sum([len(bboxes) for bboxes in pred_bboxes.values()])
                rec_cnt += len(gt_objects)
                for object_info in gt_objects:
                    pred_object_bbox = pred_bboxes[object_info['label']]
                    for bbox in pred_object_bbox:
                        iou = self.iou(object_info['bbox'], bbox)
                        if iou > self.threshold:
                            score += 1
                            break
            else:
                raise NotImplementedError
        return {
            f"mAP@{self.threshold}": (score / pre_cnt) * 100,
            f"mAR@{self.threshold}": (score / rec_cnt) * 100,
        }
            
                   


    def ppl_metric(self, answers):
        classification_score, grounding_score = 0, 0
        classification_cnt, grounding_cnt = 0, 0
        recall_cnt = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt_objects = item['gt_answers']
            pred_classification = item['classification_answer']
            pred_bboxes = item['grounding_answer']
            classification_cnt += len(pred_classification)
            recall_cnt += len(pred_bboxes)
            gt_class_names = [item['label'] for item in gt_objects]
            for class_name in pred_classification:
                if class_name in gt_class_names:
                    classification_score += 1
            for idx, object in enumerate(gt_objects):
                if object['label'] not in pred_classification:
                    continue
                grounding_cnt += 1
                pred_bbox = self.parse_func(pred_bboxes[idx])[0]
                iou = self.iou(object['bbox'], pred_bbox)
                if iou > self.threshold:
                    grounding_score +=1
        return {
            "classification_acc": (classification_score / classification_cnt) * 100,
            "grounding_acc": (grounding_score / grounding_cnt) * 100,
            f"mAP@{self.threshold}": (grounding_score / grounding_cnt) * 100,
            f"mAR@{self.threshold}": (grounding_score / recall_cnt) * 100,
        }

    def metric_func(self, answers):
        if self.inference_type == 'direct':
            return self.direct_metric(answers)
        else:
            return self.ppl_metric(answers)

class KOSMOS_Detection(Detection):
    def __init__(self, dataset_name, threshold=0.5, inference_type='direct', **kwargs):
        super().__init__(dataset_name, threshold, inference_type, **kwargs)
        from .utils import parse_kosmos_bbox
        self.parse_func = parse_kosmos_bbox