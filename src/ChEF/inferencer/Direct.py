from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import os
import datetime
from .utils import copy_batch_dict

class Direct_inferencer:

    def __init__(self,
                 dataset_name,
                 save_base_dir,
                 instruction_handler,
                 batch_size = 1,
                 max_new_tokens = 16,
                 CoT = False,
                 **kwargs) -> None:
        self.dataset_name = dataset_name
        self.save_base_dir = save_base_dir
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.CoT = CoT # whether generates CoT answer before final answer
        self.instruction_handler = instruction_handler
        self.results_path = None

    def get_collate_fn(self, dataset):
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if hasattr(dataset, 'collate'):
            collate_fn = dataset.collate
        else:
            collate_fn = lambda batch: {
                key: [data[key] for data in batch] for key in batch[0]
            }
        return collate_fn

    def inference(self, model, dataset):
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=self.get_collate_fn(dataset)
        )

        predictions = []
        for batch in tqdm(dataloader, desc="Running inference"):
            if self.CoT:
                prompts, cot = self.instruction_handler.generate_CoT_query(model, batch)
            else:
                prompts = self.instruction_handler.generate_basic_query(batch)
                cot = None
            # compatible with LAMM-style inference
            sys_msg = None if not hasattr(dataset, 'system_msg') else dataset.system_msg
            outputs = model.batch_generate(
                batch['image_path'], 
                prompts, 
                max_new_tokens=self.max_new_tokens,
                sys_msg=sys_msg,
                dataset_name=dataset.dataset_name,
                task_name=dataset.task_name,
            )
            for i in range(len(outputs)):
                answer_dict = copy_batch_dict(batch, i)
                answer_dict['query'] = prompts[i]
                answer_dict['answer'] = outputs[i]
                if self.CoT:
                    answer_dict['CoT_answer'] = cot[i]
                predictions.append(answer_dict)
        self._after_inference_step(predictions)

    def _after_inference_step(self, predictions):
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        answer_path = os.path.join(self.save_base_dir, f"{self.dataset_name}_{time}.json")
        with open(answer_path, "w") as f:
            f.write(json.dumps(predictions, indent=4))
        self.results_path = answer_path
        
del_list = ['vision_embeds_3d_ref', 'vision_embeds_3d_scene_prop','vision_pos_3d_ref','vision_pos_3d_scene_prop','mask', 'question', 'modality_embeds']        
class Direct3D_inferencer(Direct_inferencer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def inference(self, model, dataset):
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=self.get_collate_fn(dataset)
        )

        predictions = []
        for batch in tqdm(dataloader, desc="Running inference"):
            prompts = self.instruction_handler.generate_basic_query(batch)
            # compatible with LAMM-style inference
            sys_msg = None if not hasattr(dataset, 'system_msg') else dataset.system_msg
            outputs = model.batch_generate_3d(
                batch, 
                prompts, 
                sys_msg=sys_msg,
                dataset_name=dataset.dataset_name,
            )
            for i in range(len(outputs)):
                # answer_dict = copy_batch_dict(batch, i)
                answer_dict = {}
                answer_dict['query'] = prompts[i]
                answer_dict['answer'] = outputs[i]
                answer_dict['scene_id'] = batch['scene_id'][i]
                answer_dict['gt'] = batch['gt'][i]
                answer_dict['object_name'] = batch['object_name'][i]
                
                for delkey in del_list:
                    if delkey in answer_dict:
                        del answer_dict[delkey]
                predictions.append(answer_dict)
        self._after_inference_step(predictions)


class Det_Direct_inferencer(Direct_inferencer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        from metric.utils import classification_acc, Cleaner
        self.cleaner = Cleaner()
        self.check_label = classification_acc

    def inference(self, model, dataset):
        bbox_prompt = self.instruction_handler.query[1]
        predictions=[]
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

        for batch in tqdm(dataloader, desc="Running inference"):
            prompts = self.instruction_handler.generate_multiturn_query(batch, turn_idx = 0)
            outputs = model.batch_generate(batch['image_path'], prompts, max_new_tokens=self.max_new_tokens)

            res_dict_list = [dict() for i in range(len(batch['image_path']))]
            for idx, (gt_objects, pred_text) in enumerate(zip(batch['gt_answers'], outputs)):
                for gt_object in gt_objects:
                    if gt_object['label'] in res_dict_list[idx]:
                        continue
                    res_dict_list[idx][gt_object['label']] = None
                    if self.check_label(gt_object['label'], self.cleaner.clean(pred_text)): # filter correct classification answers
                        pred_bbox = model.generate(batch['image_path'][idx], bbox_prompt.format(gt_object['label']), max_new_tokens = self.max_new_tokens)
                        res_dict_list[idx][gt_object['label']] = pred_bbox
            for i in range(len(res_dict_list)):
                answer_dict = copy_batch_dict(batch, i)
                answer_dict['query'] = prompts[i] + '\n' + bbox_prompt
                answer_dict['answer'] = res_dict_list[i]
                answer_dict['classification_answer'] = outputs[i]
                predictions.append(answer_dict)

        self._after_inference_step(predictions)


class Icl_Direct_inferencer(Direct_inferencer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def inference(self, model, dataset):
        predictions=[] 
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            if self.CoT:
                prompts, cot = self.instruction_handler.generate_CoT_query(model, batch)
            else:
                prompts = self.instruction_handler.generate_basic_query(batch)

            ices = self.instruction_handler.generate_ices(prompts, batch_idx, self.batch_size)

            outputs, icl_prompts = model.icl_batch_generate(batch['image_path'], prompts, ices, self.instruction_handler.icl_cfg, max_new_tokens=self.max_new_tokens)
            for i in range(len(outputs)):
                answer_dict = copy_batch_dict(batch, i)
                answer_dict['query'] = icl_prompts[i]
                answer_dict['answer'] = outputs[i]
                if self.CoT:
                    answer_dict['CoT_answer'] = cot[i]

                predictions.append(answer_dict)
        self._after_inference_step(predictions)
