from .ice_retriever import build_retriever
from .query import build_query, build_template
import numpy as np

supported_query_types = ['standard_query', 'query_pool', 'multiturn']
class InstructionHandler:
    def __init__(self, query, answer_template, icl_cfg = None, dataset = None) -> None:
        self.query = query
        self.answer_template = answer_template
        self.icl_cfg = icl_cfg
        if icl_cfg:
            self.retriever = build_retriever(dataset, dataset, **icl_cfg)
            self.retriever.seed = icl_cfg['random_seed']
            self.ice_idx_list = self.retriever.retrieve()
            if icl_cfg['ice_with_image']:
                self.icl_cfg['use_pic'] = True
                self.icl_cfg['add_sysmsg'] = False
                self.icl_cfg['mult_conversations'] = True
            else:
                self.icl_cfg['use_pic'] = False
                self.icl_cfg['add_sysmsg'] = True
                self.icl_cfg['mult_conversations'] = False
                self.icl_cfg['sysmsg'] = 'You will now see some examples. The example has no relation to the provided image content. You need to follow the example and answer the final question based on the image content.'

    def generate_basic_query(self, batch, query=None):
        if not query:
            query = self.query
        cur_batch_len = len(batch['image_path'])
        if 'question' in batch:
            question = batch['question']
            prompts = [f'{question[i]}{query}' for i in range(cur_batch_len)]
        else:
            prompts = [query for _ in range(cur_batch_len)]
        return prompts
    
    def generate_CoT_query(self, model, batch):
        cur_batch_len = len(batch['image_path'])
        if 'question' in batch: # VQA tasks or predefined query
            question = batch['question']
            prompts = [f'{question[i]}\nLet\'s think step by step.' for i in range(cur_batch_len)]
            outputs = model.batch_generate(batch['image_path'], prompts, max_new_tokens=256)
            Lecture = outputs
            prompts_for_answer = [f'{question[i]}{self.query}' for i in range(cur_batch_len)]
            return prompts_for_answer, Lecture
        else: # not recommanded
            print('You are using CoT inferencer for neither VQA tasks nor predefined query. It is not recommanded.')
            prompts = [f'{self.query}\nLet\'s think step by step.' for i in range(cur_batch_len)]
            outputs = model.batch_generate(batch['image_path'], prompts, max_new_tokens=256)
            Lecture = outputs
            prompts_for_answer = [f'{self.query}' for i in range(cur_batch_len)]
            return prompts_for_answer, Lecture
        

    def generate_ppl_query(self, prompts, batch, batch_options, answer_template = None, ices = None, CoT = None):
        if answer_template is None:
            answer_template = self.answer_template
        '''
            if batch_option is list: ["(A) xxx", "(B) xxx"]
            if batch_option is dict: multi_turn_ppl dict(fore_label = "fore_label", options = ["(A) xxx", "(B) xxx"])
        '''
        batch_size = len(batch_options)
        if isinstance(batch_options[0], list):
            batch_ppl_len = [len(batch_option) for batch_option in batch_options]
        elif isinstance(batch_options[0], dict):
            # multi_turn ppl
            batch_ppl_len = [len(batch_option['options']) for batch_option in batch_options]
        ppl_len = sum(batch_ppl_len)
        ppl_batch_mask = np.zeros((batch_size, ppl_len))
        ppl_batch_mask_tmp_index = 0
        image_path, answers, questions, options, CoT_answer, ppl_ices = [], [], [], [], [], []

        for i in range(batch_size):
            if isinstance(batch_options[0], list):
                answers += [answer_template.format(option) for option in batch_options[i]]
                new_len = len(batch_options[i])
                questions += [prompts[i] for _ in range(new_len)]
                options += batch_options[i]
            elif isinstance(batch_options[0], dict):
                '''
                    in this case, prompts is a multi_turn prompt string
                    batch_options[i]: dict(
                        fore_label = xxx,
                        options = [a, b, c, d]
                    )
                '''
                answers += [answer_template.format(option) for option in batch_options[i]['options']]
                new_len = len(batch_options[i]['options'])
                questions += [prompts.format(batch_options[i]['fore_label']) for _ in range(new_len)]
                options += batch_options[i]['options']
            else:
                raise NotImplementedError
            image_path += [batch['image_path'][i] for _ in range(new_len)]
            ppl_batch_mask[i][ppl_batch_mask_tmp_index: ppl_batch_mask_tmp_index + new_len] = 1
            ppl_batch_mask_tmp_index += new_len
            if CoT is not None:
                CoT_answer += [CoT[i] for _ in range(new_len)]
            if ices is not None:
                ppl_ices += [ices[i] for _ in range(new_len)]
        ppl_batch_mask = np.array(ppl_batch_mask, dtype=bool)

        if CoT is not None:
            CoT = CoT_answer
        if ices is not None:
            ices = ppl_ices
        return image_path, questions, answers, ppl_batch_mask, options, CoT, ices
        
    def generate_ices(self, prompts, batch_idx, batch_size):
        ice_idx = self.ice_idx_list[batch_idx * batch_size : (batch_idx+1) * batch_size]
        ices = self.retriever.genetate_ice(ice_idx, prompts)
        return ices

    def generate_multiturn_ppl_query(self, batch, turn_idx=0, **kwargs):
        if turn_idx == 0:
            batch_size = len(batch['id'])
            return self.generate_ppl_query([self.query[0]] * batch_size, batch, answer_template=self.answer_template[0], **kwargs)
        else:
            return self.generate_ppl_query(self.query[1], batch, answer_template=self.answer_template[1], **kwargs)

    def generate_multiturn_query(self, batch, turn_idx=0, **kwargs):
        if turn_idx == 0:
            return self.generate_basic_query(batch, query=self.query[0])
        else:
            return self.generate_basic_query(batch, query=self.query[1])
        

def build_instructionhandler(task_name, 
                             dataset, 
                             query_type = 'standard_query',
                             query_assigned_ids = 0, 
                             template_assigned_ids = 0, 
                             incontext_cfg = None,
                             **kwargs):
    assert query_type in supported_query_types, f'Supported query types are {supported_query_types}, got {query_type}'

    query = build_query(task_name=task_name, query_type=query_type, assigned_ids=query_assigned_ids)
    template = build_template(task_name=task_name, assigned_ids=template_assigned_ids, query_type = query_type)
    handler = InstructionHandler(query, template, icl_cfg=incontext_cfg, dataset=dataset)
    return handler
