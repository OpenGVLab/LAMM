import os
import json
import torch
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import copy_batch_dict

class Direct_Inferencer:

    def __init__(self,
                 dataset_name,
                 save_base_dir,
                 instruction_handler,
                 dist_args,
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
        self.dist_args = dist_args

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
            predictions.extend(self.batch_inference(model, batch, dataset=dataset))
        self._after_inference_step(predictions)
    
    def generate_prompt(self, model, batch):
        if self.CoT:
            return self.instruction_handler.generate_CoT_prompt(model, batch)
        return self.instruction_handler.generate_singleturn_prompt(batch), None

    def batch_inference(self, model, batch, dataset):
        predictions = []
        prompts, cot = self.generate_prompt(model, batch)
        # compatible with LAMM-style inference
        sys_msg = None if not hasattr(dataset, 'system_msg') else dataset.system_msg
        outputs = model.batch_generate(
            batch['image_path'], 
            prompts, 
            max_new_tokens=self.max_new_tokens,
            CoT_answer_list=cot,
            sys_msg=sys_msg,
            dataset_name=dataset.dataset_name,
            task_name=dataset.task_name,
        )
        for i in range(len(outputs)):
            answer_dict = copy_batch_dict(batch, i)
            answer_dict['query'] = prompts[i]
            answer_dict['answer'] = cot[i] + outputs[i] if self.CoT else outputs[i]
            if self.CoT: answer_dict['CoT_answer'] = cot[i]
            predictions.append(answer_dict)
        return predictions

    def _after_inference_step(self, predictions):
        if self.dist_args['world_size'] == 1:
            time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            answer_path = os.path.join(self.save_base_dir, f"{self.dataset_name}_{time}.json")
        else:
            base_dir = os.path.join(self.save_base_dir, 'tmp')
            os.makedirs(base_dir, exist_ok=True)
            global_rank = self.dist_args['global_rank']
            answer_path = os.path.join(base_dir, f"{self.dataset_name}_{global_rank}.json")
        with open(answer_path, "w", encoding='utf8') as f:
            f.write(json.dumps(predictions, indent=4, ensure_ascii=False))
        self.results_path = answer_path


class PPL_Inferencer(Direct_Inferencer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def batch_inference(self, model, batch, **kwargs):
        predictions = []
        prompts, cot = self.generate_prompt(model, batch)
        batch_options = batch['options']
        return_dict = self.instruction_handler.generate_singleturn_ppl_prompt(
            prompts, batch, batch_options, CoT_answer_list = cot)
        outputs = model.ppl_inference(**return_dict)

        ppl_np = np.array(outputs)
        ppl_batch_mask = return_dict['ppl_batch_mask']
        queries = return_dict['batch_prompt']
        for idx in range(len(batch['image_path'])):
            ppl_results = ppl_np[ppl_batch_mask[idx]]
            answer_dict = copy_batch_dict(batch, idx)
            answer_dict['query'] = queries[ppl_batch_mask[idx].argmax()]
            answer_dict['ppl_results'] = ppl_results.tolist()
            if self.CoT: answer_dict['CoT_answer'] = cot[idx]
            
            score_tensor = torch.from_numpy(ppl_results)
            pred_answer_id = ppl_results.argmax()
            probs = score_tensor.softmax(dim=-1).tolist()
            answer_dict['probs'] = probs
            answer_dict['prob'] = max(probs)
            answer_dict['answer'] = batch['options'][idx][pred_answer_id]
            predictions.append(answer_dict)
        return predictions
            


class Direct3D_Inferencer(Direct_Inferencer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def batch_inference(self, model, batch, dataset):
        predictions = []
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
            answer_dict = {}
            answer_dict['query'] = prompts[i]
            answer_dict['answer'] = outputs[i]
            answer_dict['scene_id'] = batch['scene_id'][i]
            answer_dict['gt'] = batch['gt'][i]
            answer_dict['object_name'] = batch['object_name'][i]
            predictions.append(answer_dict)
        return predictions