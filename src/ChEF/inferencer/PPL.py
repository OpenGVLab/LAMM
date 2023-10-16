from .Direct import Direct_inferencer
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import copy_batch_dict
import numpy as np
import torch

class PPL_inferencer(Direct_inferencer):
    def __init__(self, calib=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calib = calib

    def inference(self, model, dataset):
        predictions=[]
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
        for batch in tqdm(dataloader, desc="Running inference"):
            if self.CoT:
                prompts, cot = self.instruction_handler.generate_CoT_query(model, batch)
            else:
                prompts = self.instruction_handler.generate_basic_query(batch)
                cot = None
            
            batch_options = batch['options']
            image_path, questions, answers, ppl_batch_mask, answer_options, CoT_answer, _ = self.instruction_handler.generate_ppl_query(prompts, batch, batch_options, CoT = cot)
            if self.calib:
                outputs = model.cali_inference(image_path, questions, answers, answer_options, CoT_answer)
            else:
                outputs = model.ppl_inference(image_path, questions, answers, answer_options, CoT_answer)

            ppl_np = np.array(outputs)
            for idx in range(len(batch['image_path'])):
                ppl_results = ppl_np[ppl_batch_mask[idx]]
                answer_dict = copy_batch_dict(batch, idx)
                answer_dict['query'] = questions[ppl_batch_mask[idx].argmax()]
                answer_dict['ppl_results'] = ppl_results.tolist()
                if self.CoT:
                    answer_dict['CoT_answer'] = cot[idx]

                if self.calib:
                    score_tensor = torch.from_numpy(ppl_results)
                    pred_answer_id = ppl_results.argmax()
                    probs = score_tensor.softmax(dim=-1).tolist()
                    answer_dict['probs'] = probs
                    answer_dict['prob'] = max(probs)
                else:
                    pred_answer_id = ppl_results.argmin()
                answer_dict['answer'] = batch['options'][idx][pred_answer_id]
                predictions.append(answer_dict)

        self._after_inference_step(predictions)


class ICL_PPL_inferencer(Direct_inferencer):
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
                cot = None
            ices = self.instruction_handler.generate_ices(prompts, batch_idx, self.batch_size)
            
            batch_options = batch['options']
            image_path, questions, answers, ppl_batch_mask, answer_options, CoT_answer, ices = self.instruction_handler.generate_ppl_query(prompts, batch, batch_options, ices, CoT = cot)
            outputs, icl_prompts = model.icl_ppl_inference(image_path, questions, answers, answer_options, ices, self.instruction_handler.icl_cfg, CoT_answer)
            ppl_np = np.array(outputs)
            icl_prompt_idx = 0
            for idx in range(len(batch['id'])):
                ppl_results = ppl_np[ppl_batch_mask[idx]]
                pred_answer_id = ppl_results.argmin()
                answer_dict = copy_batch_dict(batch, idx)
                answer_dict['query'] = icl_prompts[icl_prompt_idx]
                answer_dict['ppl_results'] = ppl_results.tolist()
                if self.CoT:
                    answer_dict['CoT_answer'] = cot[idx]
                answer_dict['answer'] = batch['options'][idx][pred_answer_id]
                predictions.append(answer_dict)
                icl_prompt_idx += len(ppl_results)

        self._after_inference_step(predictions)


class Det_PPL_inferencer(Direct_inferencer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def inference(self, model, dataset):
        predictions=[]
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
        for batch in tqdm(dataloader, desc="Running inference"):
            cur_batch_len = len(batch['image_path'])
            
            classification_num_turns = max([len(options) for options in batch['classification_options']])
            classification_ppl_list = []
            classification_ppl_batch_mask_list = []
            for i in range(classification_num_turns):
                batch_options = [(options[i] if len(options)>i else []) for options in batch['classification_options']]
                image_path, cls_questions, answers, ppl_batch_mask, answer_options, _, _ = self.instruction_handler.generate_multiturn_ppl_query(batch, turn_idx = 0, batch_options = batch_options)
                outputs = model.ppl_inference(image_path, cls_questions, answers, answer_options)
                classification_ppl_list.append(np.array(outputs))
                classification_ppl_batch_mask_list.append(ppl_batch_mask)

            grounding_num_turns = max([len(options) for options in batch['grounding_options']])
            grounding_ppl_list = []
            grounding_ppl_batch_mask_list = []
            for i in range(grounding_num_turns):
                batch_options = [(options[i] if len(options)>i else dict(fore_label = None, options = []) ) for options in batch['grounding_options']]
                image_path, grd_questions, answers, ppl_batch_mask, answer_options, _, _ = self.instruction_handler.generate_multiturn_ppl_query(batch, turn_idx = 1, batch_options = batch_options)
                outputs = model.ppl_inference(image_path, grd_questions, answers, answer_options)
                grounding_ppl_list.append(np.array(outputs))
                grounding_ppl_batch_mask_list.append(ppl_batch_mask)

            for idx in range(cur_batch_len):
                answer_dict = copy_batch_dict(batch, idx)
                answer_dict['query'] = cls_questions[ppl_batch_mask[idx].argmax()] + '\n' + grd_questions[ppl_batch_mask[idx].argmax()]
                classification_ppl_results = [ppl_np[ppl_batch_mask[idx]] for ppl_np, ppl_batch_mask in zip(classification_ppl_list, classification_ppl_batch_mask_list)]
                classification_ppl_results = [result for result in classification_ppl_results if len(result) > 0]
                pred_answer_id_list = [ppl_result.argmin() for ppl_result in classification_ppl_results]
                answer_dict['classification_ppl_results'] = [ppl_result.tolist() for ppl_result in classification_ppl_results]
                answer_dict['classification_answer'] = [batch['classification_options'][idx][id][pred_answer_id] for (id, pred_answer_id) in enumerate(pred_answer_id_list)]
                
                grounding_ppl_results = [ppl_np[ppl_batch_mask[idx]] for ppl_np, ppl_batch_mask in zip(grounding_ppl_list, grounding_ppl_batch_mask_list)]
                grounding_ppl_results = [result for result in grounding_ppl_results if len(result) > 0]
                pred_answer_id_list = [ppl_result.argmin() for ppl_result in grounding_ppl_results]
                answer_dict['grounding_ppl_results'] = [ppl_result.tolist() for ppl_result in grounding_ppl_results]
                answer_dict['grounding_answer'] = [batch['grounding_options'][idx][id]['options'][pred_answer_id] for (id, pred_answer_id) in enumerate(pred_answer_id_list)]
                predictions.append(answer_dict)

        self._after_inference_step(predictions)