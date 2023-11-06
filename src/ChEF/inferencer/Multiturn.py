from .Direct import Direct_inferencer
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import copy_batch_dict
import numpy as np

class Multi_Turn_PPL_inferencer(Direct_inferencer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def inference(self, model, dataset):
        predictions=[]
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
        for batch in tqdm(dataloader, desc="Running inference"):
            cur_batch_len = len(batch['image_path'])
            
            # first turn
            batch_options = [options[0]['options'] for options in batch['options']]
            image_path, questionsa, answers, ppl_batch_mask, answer_options, _, _ = self.instruction_handler.generate_multiturn_ppl_query(batch = batch, turn_idx = 0, batch_options = batch_options)

            outputs = model.ppl_inference(image_path, questionsa, answers, answer_options)
            ppl_np = np.array(outputs)

            ppl_np_list = [ppl_np]
            ppl_batch_mask_list = [ppl_batch_mask]

            num_turns = len(batch['options'][0])
            for i in range(1, num_turns, 1):
                batch_options = [options[i] for options in batch['options']]
                image_path, questionsb, answers, ppl_batch_mask, answer_options, _, _ = self.instruction_handler.generate_multiturn_ppl_query(batch = batch, turn_idx = i, batch_options = batch_options)
                outputs = model.ppl_inference(image_path, questionsb, answers, answer_options)
                ppl_np = np.array(outputs)
                ppl_np_list.append(ppl_np)
                ppl_batch_mask_list.append(ppl_batch_mask)

            for idx in range(cur_batch_len):
                ppl_results = [ppl_np[ppl_batch_mask[idx]] for ppl_np, ppl_batch_mask in zip(ppl_np_list, ppl_batch_mask_list)]
                pred_answer_id_list = [ppl_result.argmin() for ppl_result in ppl_results]
                answer_dict = copy_batch_dict(batch, idx)
                answer_dict['query'] = self.instruction_handler.query
                answer_dict['ppl_results'] = [ppl_result.tolist() for ppl_result in ppl_results]
                answer_dict['answer'] = [batch['options'][idx][id]['options'][pred_answer_id] for (id, pred_answer_id) in enumerate(pred_answer_id_list)]
                predictions.append(answer_dict)

        self._after_inference_step(predictions)