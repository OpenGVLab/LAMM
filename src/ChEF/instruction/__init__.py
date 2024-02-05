from .ice_retriever import build_retriever
from .prompt import build_prompt
from .template import build_template
import numpy as np

def get_cur_batch_len(batch):
    if 'id' in batch:
        cur_batch_len = len(batch['id'])
    elif 'image_path' in batch:
        cur_batch_len = len(batch['image_path'])
    elif 'pcl_paths' in batch:
        cur_batch_len = len(batch['pcl_paths'])
    elif 'task_type' in batch:
        cur_batch_len = len(batch['task_type'])
    else:
        raise ValueError('cannot get batch size')
    return cur_batch_len

supported_prompt_types = ['singleturn', 'multiturn']
class InstructionHandler:
    def __init__(self, prompt, answer_template, incontext_cfg=None, dataset=None):
        self.prompt = prompt
        self.CoT_prompt = 'Let\'s think step by step.'
        self.answer_template = answer_template
        self.incontext_cfg = incontext_cfg
        if incontext_cfg:
            self.retriever = build_retriever(dataset, dataset, **incontext_cfg)
            self.retriever.seed = incontext_cfg['random_seed']
            self.ice_idx_list = self.retriever.retrieve()

    def _query_format(self, prompt, question): # TODO: add icl
        if '{question}' in prompt:
            return prompt.format(question=question)
        assert question == '', f'Need question formatted in prompt, but \"{prompt}\" does not support.'
        return prompt
            
    def generate_singleturn_prompt(self, batch):
        prompt = self.prompt
        cur_batch_len = get_cur_batch_len(batch)

        question_list = batch['question'] if 'question' in batch \
            else [''] * cur_batch_len
        query = [self._query_format(prompt, question) for question in question_list]
        return query
    
    def generate_CoT_prompt(self, model, batch, max_new_tokens=256):
        cur_batch_len = get_cur_batch_len(batch)
        if 'question' in batch: # VQA tasks or predefined prompt
            question_list = batch['question']
            query_for_CoT = [f'{question}\n{self.CoT_prompt}' for question in question_list]
            CoT_response = model.batch_generate(batch['image_path'], query_for_CoT, max_new_tokens=max_new_tokens)
            query_for_answer = [self._query_format(self.prompt, question) for question in question_list]
        else: # not recommanded
            print('You are using CoT inferencer for neither VQA tasks nor predefined prompt. It is not recommanded.')
            query_for_CoT = [f'{self.prompt}\n{self.CoT_prompt}' for i in range(cur_batch_len)]
            CoT_response = model.batch_generate(batch['image_path'], query_for_CoT, max_new_tokens=max_new_tokens)
            query_for_answer = [f'{self.prompt}' for i in range(cur_batch_len)]
        return query_for_answer, CoT_response

    def generate_singleturn_ppl_prompt(self, prompts, batch, batch_options, **kwargs):
        answer_template = self.answer_template
        batch_size = len(batch_options)
        batch_ppl_len = [len(batch_option) for batch_option in batch_options]
        ppl_len = sum(batch_ppl_len)
        ppl_batch_mask = np.zeros((batch_size, ppl_len))
        ppl_batch_mask_tmp_index = 0
        image_path, answers, questions, options= [], [], [], []
        return_dict = {key: [] for key in kwargs.keys()}

        for i in range(batch_size):
            answers += [answer_template.format(option=option) for option in batch_options[i]]
            options += batch_options[i]
            new_len = len(batch_options[i])
            questions += [prompts[i]] * new_len
            image_path += [batch['image_path'][i]] * new_len
            ppl_batch_mask[i][ppl_batch_mask_tmp_index: ppl_batch_mask_tmp_index + new_len] = 1
            ppl_batch_mask_tmp_index += new_len
            for key in return_dict.keys():
                return_dict[key] += [kwargs[key][i]] * new_len
        
        ppl_batch_mask = np.array(ppl_batch_mask, dtype=bool)
        return_dict['batch_images'] = image_path
        return_dict['batch_prompt'] = questions
        return_dict['batch_answers'] = answers
        return_dict['batch_options'] = options
        return_dict['ppl_batch_mask'] = ppl_batch_mask
        return return_dict
        
    def generate_ices(self, prompts, batch_idx, batch_size):
        ice_idx = self.ice_idx_list[batch_idx * batch_size : (batch_idx+1) * batch_size]
        ices = self.retriever.genetate_ice(ice_idx, prompts)
        return ices

    def generate_multiturn_ppl_prompt(self, batch, prompt_idx_list, prefix_list, batch_options, **kwargs):
        prompt_idx_list = [min(prompt_idx,1) if prompt_idx is not None \
            else None for prompt_idx in prompt_idx_list]
        prompt_list = []
        for prompt_idx, prefix in zip(prompt_idx_list, prefix_list):
            if prompt_idx is None:
                prompt_list.append(None)
                continue
            prompt_list.append(self.prompt[prompt_idx].format(prefix=prefix))
        answer_template_list = [self.answer_template[prompt_idx] \
            if prompt_idx is not None else None \
                for prompt_idx in prompt_idx_list]

        multi_turn_batch_index = []
        multi_turn_batch_tmp_index = 0
        
        image_path, answers, questions, options= [], [], [], []
        return_dict = {key: [] for key in kwargs.keys()}
        
        for i, (prompt, answer_template, sample_option) in enumerate(zip(prompt_list,answer_template_list, batch_options)):
            if prompt is None: # sample nothing to inference in this turn 
                multi_turn_batch_index.append(None)
                continue
            answers += [answer_template.format(option=option) for option in sample_option]
            options += sample_option
            new_len = len(sample_option)
            questions += [prompt] * new_len
            image_path += [batch['image_path'][i]] * new_len
            multi_turn_batch_index.append([i for i in range(multi_turn_batch_tmp_index, multi_turn_batch_tmp_index + new_len)])
            multi_turn_batch_tmp_index += new_len
            for key in return_dict.keys():
                return_dict[key] += [kwargs[key][i]] * new_len
        return_dict['batch_images'] = image_path
        return_dict['batch_prompt'] = questions
        return_dict['batch_answers'] = answers
        return_dict['batch_options'] = options
        return_dict['ppl_batch_index'] = multi_turn_batch_index
        return return_dict

    def generate_multiturn_prompt(self, batch, prompt_idx_list, prefix_list, **kwargs):
        prompt_idx_list = [min(prompt_idx,1) if prompt_idx is not None \
            else None for prompt_idx in prompt_idx_list]
        multi_turn_batch_index = []
        multi_turn_batch_tmp_index = 0
        image_path, questions = [], []
        return_dict = {key: [] for key in kwargs.keys()}

        for i, (prompt_idx, prefix) in enumerate(zip(prompt_idx_list, prefix_list)):
            if prompt_idx is None:
                multi_turn_batch_index.append(None)
                continue
            image_path.append(batch['image_path'][i])
            questions.append(self.prompt[prompt_idx].format(prefix=prefix))
            multi_turn_batch_index.append(multi_turn_batch_tmp_index)
            multi_turn_batch_tmp_index += 1
            for key in return_dict.keys():
                return_dict[key].append(kwargs[key][i])
        return_dict['batch_images'] = image_path
        return_dict['batch_prompt'] = questions
        return_dict['multi_turn_batch_index'] = multi_turn_batch_index
        return return_dict
        

def build_instructionhandler(
    task_name, 
    dataset, 
    prompt_type = 'singleturn',
    prompt_assigned_ids = 0, 
    template_assigned_ids = 0, 
    incontext_cfg = None,
    **kwargs):
    assert prompt_type in supported_prompt_types, f'Supported prompt types are {supported_prompt_types}, got {prompt_type}'
    prompt = build_prompt(
        task_name=task_name, 
        prompt_type=prompt_type, 
        assigned_ids=prompt_assigned_ids, 
        **kwargs)
    template = build_template(
        task_name=task_name, 
        assigned_ids=template_assigned_ids, 
        prompt_type=prompt_type,
        **kwargs)
    handler = InstructionHandler(
        prompt, 
        template, 
        incontext_cfg=incontext_cfg, 
        dataset=dataset)
    return handler
