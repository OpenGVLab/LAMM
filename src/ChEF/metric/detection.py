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
