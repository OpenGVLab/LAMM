import torch
import numpy as np
from .utils import copy_batch_dict
from .Singleturn import Direct_Inferencer

class Multi_Turn_PPL_Inferencer(Direct_Inferencer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def batch_inference(self, model, batch, **kwargs):
        multiturn_prefix = batch['multi_turn_prefix']
        turn_num = max([len(item) for item in multiturn_prefix])
        predictions = [copy_batch_dict(batch, i) \
            for i in range(len(multiturn_prefix))]
        for item in predictions: item['turn_answer'] = []

        for turn_idx in range(turn_num):
            prompt_idx_list = [item_prefix[turn_idx]['prompt_idx'] \
                if len(item_prefix)>turn_idx else None \
                    for item_prefix in multiturn_prefix]
            prefix_list =  [item_prefix[turn_idx]['prefix'] \
                if len(item_prefix)>turn_idx else None \
                    for item_prefix in multiturn_prefix]
            batch_options = [options[turn_idx]['options'] \
                if len(options) > turn_idx else None \
                    for options in batch['options']]
            multiturn_prompt_info = dict(
                batch=batch, 
                prompt_idx_list=prompt_idx_list, 
                prefix_list=prefix_list, 
                batch_options=batch_options
            )

            return_dict = self.instruction_handler.generate_multiturn_ppl_prompt(**multiturn_prompt_info)
            outputs = model.ppl_inference(**return_dict)
            ppl_np = np.array(outputs)
            ppl_batch_index = return_dict['ppl_batch_index']
            for idx in range(len(batch['image_path'])):
                if ppl_batch_index[idx] is None:
                    continue
                ppl_results = ppl_np[[i for i in ppl_batch_index[idx]]]

                pred_answer_id = ppl_results.argmax()
                score_tensor = torch.from_numpy(ppl_results)
                probs = score_tensor.softmax(dim=-1).tolist()
                predictions[idx]['turn_answer'].append(dict(
                    prompt_idx = prompt_idx_list[idx],
                    question = return_dict['batch_prompt'][ppl_batch_index[idx][0]],
                    answer = batch_options[idx][pred_answer_id],
                    probs = probs,
                    prob = max(probs),
                    options = batch_options[idx]
                ))
        return predictions
    

class Multi_Direct_Inferencer(Direct_Inferencer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def batch_inference(self, model, batch, **kwargs):
        multiturn_prefix = batch['multi_turn_prefix']
        turn_num = max([len(item) for item in multiturn_prefix])
        predictions = [copy_batch_dict(batch, i) \
            for i in range(len(multiturn_prefix))]
        for item in predictions: item['turn_answer'] = []
        
        for turn_idx in range(turn_num):
            prompt_idx_list = [item_prefix[turn_idx]['prompt_idx'] \
                if len(item_prefix)>turn_idx else None \
                    for item_prefix in multiturn_prefix]
            prefix_list =  [item_prefix[turn_idx]['prefix'] \
                if len(item_prefix)>turn_idx else None \
                    for item_prefix in multiturn_prefix]
            return_dict = self.instruction_handler.generate_multiturn_prompt(
                batch, prompt_idx_list, prefix_list)
            outputs = model.batch_generate(
                max_new_tokens=self.max_new_tokens,
                **return_dict
            )
            multi_turn_batch_index = return_dict['multi_turn_batch_index']
            for i in range(len(multiturn_prefix)):
                answer_index = multi_turn_batch_index[i]
                if answer_index is None:
                    continue
                predictions[i]['turn_answer'].append(dict(
                    prompt_idx = prompt_idx_list[i],
                    question = return_dict['batch_prompt'][answer_index],
                    answer = outputs[answer_index],
                ))

        return predictions