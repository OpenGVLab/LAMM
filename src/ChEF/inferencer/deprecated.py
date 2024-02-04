
# from .deprecated import Direct_Inferencer
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from .utils import copy_batch_dict
# import numpy as np
# import torch

# class ICL_PPL_Inferencer(Direct_Inferencer):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#     def inference(self, model, dataset):
#         predictions=[]
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

#         for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
#             if self.CoT:
#                 prompts, cot = self.instruction_handler.generate_CoT_query(model, batch)
#             else:
#                 prompts = self.instruction_handler.generate_basic_query(batch)
#                 cot = None
#             ices = self.instruction_handler.generate_ices(prompts, batch_idx, self.batch_size)
            
#             batch_options = batch['options']
#             image_path, questions, answers, ppl_batch_mask, answer_options, CoT_answer, ices = self.instruction_handler.generate_ppl_query(prompts, batch, batch_options, ices = ices, CoT = cot)
#             outputs, icl_prompts = model.icl_ppl_inference(image_path, questions, answers, answer_options, ices, self.instruction_handler.icl_cfg, CoT_answer)
#             ppl_np = np.array(outputs)
#             icl_prompt_idx = 0
#             for idx in range(len(batch['id'])):
#                 ppl_results = ppl_np[ppl_batch_mask[idx]]
#                 pred_answer_id = ppl_results.argmin()
#                 answer_dict = copy_batch_dict(batch, idx)
#                 answer_dict['query'] = icl_prompts[icl_prompt_idx]
#                 answer_dict['ppl_results'] = ppl_results.tolist()
#                 answer_dict['ices'] = ices[icl_prompt_idx]
#                 if ices[icl_prompt_idx] !=[] and not isinstance(ices[icl_prompt_idx][0]['image_path'], str):
#                     for i in range(len(ices[icl_prompt_idx])):
#                         del answer_dict['ices'][i]['image_path']
#                 if self.CoT:
#                     answer_dict['CoT_answer'] = cot[idx]
#                 answer_dict['answer'] = batch['options'][idx][pred_answer_id]
#                 predictions.append(answer_dict)
#                 icl_prompt_idx += len(ppl_results)

#         self._after_inference_step(predictions)


# class Det_PPL_Inferencer(Direct_Inferencer):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#     def inference(self, model, dataset):
#         predictions=[]
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
#         for batch in tqdm(dataloader, desc="Running inference"):
#             cur_batch_len = len(batch['image_path'])
            
#             classification_num_turns = max([len(options) for options in batch['classification_options']])
#             classification_ppl_list = []
#             classification_ppl_batch_mask_list = []
#             for i in range(classification_num_turns):
#                 batch_options = [(options[i] if len(options)>i else []) for options in batch['classification_options']]
#                 image_path, cls_questions, answers, ppl_batch_mask, answer_options, _, _ = self.instruction_handler.generate_multiturn_ppl_query(batch, turn_idx = 0, batch_options = batch_options)
#                 import ipdb;ipdb.set_trace()
#                 outputs = model.ppl_inference(image_path, cls_questions, answers, answer_options)
#                 classification_ppl_list.append(np.array(outputs))
#                 classification_ppl_batch_mask_list.append(ppl_batch_mask)

#             grounding_num_turns = max([len(options) for options in batch['grounding_options']])
#             grounding_ppl_list = []
#             grounding_ppl_batch_mask_list = []
#             for i in range(grounding_num_turns):
#                 batch_options = [(options[i] if len(options)>i else dict(fore_label = None, options = []) ) for options in batch['grounding_options']]
#                 image_path, grd_questions, answers, ppl_batch_mask, answer_options, _, _ = self.instruction_handler.generate_multiturn_ppl_query(batch, turn_idx = 1, batch_options = batch_options)
#                 outputs = model.ppl_inference(image_path, grd_questions, answers, answer_options)
#                 grounding_ppl_list.append(np.array(outputs))
#                 grounding_ppl_batch_mask_list.append(ppl_batch_mask)

#             for idx in range(cur_batch_len):
#                 answer_dict = copy_batch_dict(batch, idx)
#                 answer_dict['query'] = cls_questions[0] + '\n' + grd_questions[0]
#                 classification_ppl_results = [ppl_np[ppl_batch_mask[idx]] for ppl_np, ppl_batch_mask in zip(classification_ppl_list, classification_ppl_batch_mask_list)]
#                 classification_ppl_results = [result for result in classification_ppl_results if len(result) > 0]
#                 pred_answer_id_list = [ppl_result.argmin() for ppl_result in classification_ppl_results]
#                 answer_dict['classification_ppl_results'] = [ppl_result.tolist() for ppl_result in classification_ppl_results]
#                 answer_dict['classification_answer'] = [batch['classification_options'][idx][id][pred_answer_id] for (id, pred_answer_id) in enumerate(pred_answer_id_list)]
                
#                 grounding_ppl_results = [ppl_np[ppl_batch_mask[idx]] for ppl_np, ppl_batch_mask in zip(grounding_ppl_list, grounding_ppl_batch_mask_list)]
#                 grounding_ppl_results = [result for result in grounding_ppl_results if len(result) > 0]
#                 pred_answer_id_list = [ppl_result.argmin() for ppl_result in grounding_ppl_results]
#                 answer_dict['grounding_ppl_results'] = [ppl_result.tolist() for ppl_result in grounding_ppl_results]
#                 answer_dict['grounding_answer'] = [batch['grounding_options'][idx][id]['options'][pred_answer_id] for (id, pred_answer_id) in enumerate(pred_answer_id_list)]
#                 predictions.append(answer_dict)

#         self._after_inference_step(predictions)

    
# class Cali_Inferencer(Direct_Inferencer):
#     def __init__(self,  **kwargs) -> None:
#         super().__init__(**kwargs)

#     def inference(self, model, dataset):
#         predictions=[]
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
#         for batch in tqdm(dataloader, desc="Running inference"):
#             cur_batch_len = len(batch['image_path'])
#             if self.CoT:
#                 prompts, cot = self.instruction_handler.generate_CoT_query(model, batch)
#             else:
#                 prompts = self.instruction_handler.generate_basic_query(batch)
#                 cot=None
            
#             batch_options = batch['options']
#             image_path, questions, answers, ppl_batch_mask, answer_options, CoT_answer, _ = self.instruction_handler.generate_ppl_query(prompts, batch, batch_options, CoT = cot)
#             score = model.do_calibration(image_path, questions, answers, answer_options, CoT_answer)
#             score_np = np.array(score)
#             for idx in range(cur_batch_len):
#                 score_results = score_np[ppl_batch_mask[idx]]
#                 score_tensor = torch.from_numpy(score_results)
#                 pred_answer_id = score_results.argmax()
#                 answer_dict = copy_batch_dict(batch, idx)
#                 answer_dict['query'] = prompts[idx]
#                 answer_dict['ppl_results'] = score_results.tolist()
#                 if self.CoT:
#                     answer_dict['CoT_answer'] = cot[idx]
#                 answer_dict['answer'] = batch['options'][idx][pred_answer_id]
#                 probs = score_tensor.softmax(dim=-1).tolist()
#                 answer_dict['probs'] = probs
#                 answer_dict['prob'] = max(probs)
#                 predictions.append(answer_dict)
#         self._after_inference_step(predictions)