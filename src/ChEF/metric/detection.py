import re
import torch
from torchvision.ops import box_iou
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
            rec_cnt += len(gt_objects)
            if 'turn_answer' in item: # multiturn
                detection_turn_info = item['multi_turn_prefix'][1:]
                classification_answer = item['turn_answer'][0]['answer']
                detection_turn_answer = [answer['answer'] for answer in item['turn_answer'][1:]]
                correct_pred_objects = []
                for object_info in gt_objects:
                    if self.check_func(object_info['label'], classification_answer):
                        correct_pred_objects.append(object_info['label'])
                        break
                gt_label_bbox = {key: [gt_object['bbox'] for gt_object in gt_objects \
                    if gt_object['label'] == key] \
                        for key in correct_pred_objects}

                for turn_info, turn_answer in zip(detection_turn_info, detection_turn_answer):
                    if turn_info['prefix'] in correct_pred_objects:
                        pred_bbox_list = self.parse_func(turn_answer)
                        gt_bbox_list = gt_label_bbox[turn_info['prefix']]
                        pre_cnt += len(pred_bbox_list)

                        for pred_bbox in pred_bbox_list:
                            for gt_bbox in gt_bbox_list:
                                iou = self.iou(gt_bbox, pred_bbox)
                                if iou > self.threshold:
                                    score += 1
                                    break
            else: # singleturn
                text = item['answer']
                assert isinstance(text, str)
                bboxes = self.parse_func(text)
                pre_cnt += len(bboxes)
                for object_info in gt_objects:
                    if not self.check_func(object_info['label'], text):
                        continue
                    for bbox in bboxes:
                        iou = self.iou(object_info['bbox'], bbox)
                        if iou > self.threshold:
                            score += 1
                            break
        return {
            f"mAP@{self.threshold}": (score / pre_cnt) * 100,
            f"mAR@{self.threshold}": (score / rec_cnt) * 100,
        }

    def ppl_metric(self, answers):
        classification_score, grounding_score = 0, 0
        classification_cnt, grounding_cnt = 0, 0
        recall_cnt = 0
        for item in tqdm(answers, desc="Running Metric"): # multiturn answer
            gt_objects = item['gt_answers']
            recall_cnt += len(gt_objects)
            turn_answers = item['turn_answer']
            pred_classification = []
            for turn_answer in turn_answers:
                if turn_answer['prompt_idx'] == 0:
                    pred_classification.append(turn_answer['answer'])

            classification_cnt += len(pred_classification)

            gt_class_names = set([item['label'] for item in gt_objects])
            correct_pred_objects = []
            for class_name in pred_classification:
                for gt_class_name in gt_class_names:
                    if self.check_func(gt_class_name, class_name):
                        correct_pred_objects.append(gt_class_name)
                        classification_score += 1
                        break
            
            multi_turn_prefix = item['multi_turn_prefix']
            for turn_answer, prefix_info in zip(turn_answers, multi_turn_prefix):
                if turn_answer['prompt_idx'] == 1 and prefix_info['prefix'] in correct_pred_objects:
                    grounding_cnt += 1
                    for object in gt_objects:
                        if object['label'] != prefix_info['prefix']:
                            continue
                        pred_bbox = self.parse_func(turn_answer['answer'])[0]
                        iou = self.iou(object['bbox'], pred_bbox)
                        if iou > self.threshold:
                            grounding_score +=1
                            break
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

class LAMM_Detection(Base_Metric):

    def __init__(self, dataset_name, iou_thres=[0.5, 0.25]):
        super().__init__(dataset_name)
        self.iou_thres = iou_thres

    def hungarian_match(
        self,
        gt_bboxes, 
        pred_bboxes, 
        iou_matrix,
        iou_thres
    ):
        match_id = torch.full((len(gt_bboxes),), -1)
        vis = torch.zeros(len(gt_bboxes))

        def dfs(pred_i):
            for gt_i in range(len(gt_bboxes)):
                if iou_matrix[gt_i, pred_i] >= iou_thres and vis[gt_i] == 0:
                    vis[gt_i] = 1
                    if match_id[gt_i] == -1 or dfs(match_id[gt_i]):
                        match_id[gt_i] = pred_i
                        return True
            return False

        for pred_i in range(len(pred_bboxes)):
            vis = torch.zeros(len(gt_bboxes))
            dfs(pred_i)
        return match_id

    def hungarian_match_with_class(
        self,
        gt_bboxes, 
        gt_labels,
        pred_bboxes, 
        pred_labels,
        iou_matrix, 
        iou_thres
    ):
        from .utils import classification_acc
        for gt_i, gt_label in enumerate(gt_labels):
            for pred_i, pred_label in enumerate(pred_labels):
                result = classification_acc(gt_label, pred_label)
                if result is False:
                    iou_matrix[gt_i, pred_i] *= 0.

        return self.hungarian_match(gt_bboxes, pred_bboxes, iou_matrix, iou_thres)

    def parser(self, text, ignore_coord_order=False):
        pat_cat = [
            r'The ([\w\W ]+?)(?=object)',
            r'classified as ([\w\W ]+?)(?= is| and| within|\.|,)',
            r'categorized as ([\w\W ]+?)(?= can| at|\.|,)',
            r'belong(?:s|ing) to the category of ([\w\W ]+?)(?= at|\.|,)',
            r'belong(?:s|ing) to the ([\w\W ]+) category',
            r'falls under the category of ([\w\W ]+?)(?=\.|,)',
            r'falls under the ([\w\W ]+) category',
            r'its category is ([\w\W ]+?)(?=\.|,)',
        ]
        pat_bbox = r'\[ ?([\d\.]+), ?([\d\.]+), ?([\d\.]+), ?([\d\.]+) ?\]'

        boxes = text.strip().split('. ')
        ret_boxes, ret_cls = [], []
        for i, box in enumerate(boxes):
            box = box.strip()
            if len(box) == 0:
                continue
            if i == len(boxes) - 1 and box[-1] == '.':
                box = box[:-1]
            box += '.'

            res_bbox = re.search(pat_bbox, box)
            if res_bbox is None:
                continue
            res_cat = None
            for pat_cat_i in pat_cat:
                res = re.search(pat_cat_i, box)
                if res is not None:
                    res_cat = res
            if res_cat is None:
                continue

            x1 = float(res_bbox.group(1))
            y1 = float(res_bbox.group(2))
            x2 = float(res_bbox.group(3))
            y2 = float(res_bbox.group(4))
            category = res_cat.group(1).strip()

            if x1 <= x2 and y1 <= y2:
                ret_boxes.append([x1, y1, x2, y2])
                ret_cls.append(category)
            else:
                if ignore_coord_order and x1 > x2 and y1 > y2:
                    ret_boxes.append([x2, y2, x1, y1])
                    ret_cls.append(category)
                else:
                    continue

        return ret_boxes, ret_cls

    def metric_func(self, answers):
        tp = [0 for _ in range(len(self.iou_thres))]
        tp_with_cls = [0 for _ in range(len(self.iou_thres))]
        num_pred, num_gt = 0, 0

        for item in tqdm(answers, desc='Running Metric'):
            gt_objects = item['gt_answers']
            gt_bboxes = torch.tensor([gt_object['bbox'] for gt_object in gt_objects])
            gt_labels = [gt_object['label'] for gt_object in gt_objects]
            num_gt += len(gt_objects)

            text = item['answer']
            pred_bboxes, pred_labels = self.parser(text)
            pred_bboxes = torch.tensor(pred_bboxes)
            if len(pred_bboxes) == 0:
                continue
            num_pred += len(pred_bboxes)

            iou_matrix = box_iou(gt_bboxes, pred_bboxes)
            for iou_i, iou_thres in enumerate(self.iou_thres):
                gt_match_id = self.hungarian_match(
                    gt_bboxes, pred_bboxes, iou_matrix.clone(), iou_thres)
                tp[iou_i] += (gt_match_id != -1).sum()

                gt_match_id = self.hungarian_match_with_class(
                    gt_bboxes, gt_labels, pred_bboxes, pred_labels, iou_matrix.clone(), iou_thres)
                tp_with_cls[iou_i] += (gt_match_id != -1).sum()
        
        metric_dict = dict()
        for iou_i, iou_thres in enumerate(self.iou_thres):
            metric_dict.update({
                f'recall@{iou_thres:.2f}': (tp_with_cls[iou_i] / num_gt * 100).item(),
                f'prec@{iou_thres:.2f}': (tp_with_cls[iou_i] / (num_pred + 1e-7) * 100).item(),
                f'recall_wocat@{iou_thres:.2f}': (tp[iou_i] / num_gt * 100).item(),
                f'prec_wocat@{iou_thres:.2f}': (tp[iou_i] / (num_pred + 1e-7) * 100).item(),
            })
        return metric_dict

class LAMM_3D_Detection(Base_Metric):

    def __init__(self, dataset_name, thres = 0.5,  **kwargs):
        super().__init__(dataset_name)
        self.thres = thres
        from .utils import parse_bbox_3d, classification_acc, cal_iou_3d
        self.parse = parse_bbox_3d
        self.cls = classification_acc
        self.iou = cal_iou_3d

    def metric_func(self, answers):
        score = 0.0
        cnt = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt_objects = item['gt_answers']
            text = item['answer']
            bboxes = self.parse(text)
            cnt += len(gt_objects)
            for object_info in gt_objects:
                if not self.cls(object_info['label'], text):
                    continue
                for bbox in bboxes:
                    iou = self.iou(object_info['bbox'], bbox)
                    if iou > self.thres:
                        score += 1
                        break
        return dict(
            mAR = score/cnt * 100,
        )

class LAMM_3D_Grounding(Base_Metric):

    def __init__(self, dataset_name, thres = 0.5,  **kwargs):
        super().__init__(dataset_name)
        self.thres = thres
        from .utils import parse_bbox_3d, cal_iou_3d
        self.parse = parse_bbox_3d
        self.iou = cal_iou_3d

    def metric_func(self, answers):
        score = 0.0
        cnt = 0
        for item in tqdm(answers, desc="Running Metric"):
            gtobject = item['gt_answers']
            text = item['answer']
            bboxes = self.parse(text)
            cnt += 1
            if len(bboxes) < 1:
                continue
            bbox = bboxes[0]
            iou = self.iou(gtobject, bbox)
            if iou > self.thres:
                score += 1
        return dict(
            mAR = score/cnt * 100,
        )
