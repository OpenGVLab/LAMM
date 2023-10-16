import os
import json
from torch.utils.data import Dataset
import random
class FlickrDataset(Dataset):
    task_name = 'caption'
    dataset_name = 'Flickr30k'
    def __init__(self, 
                 base_data_path = 'data/datasets/LAMM//2D_Benchmark/',
                 ppl_cfg = None,
                 **kwargs):
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,'meta_file', 'Caption_flickr30k.json')
        self.data = json.load(open(json_path,'rb'))
        self.ppl_cfg = ppl_cfg
        self.ppl = False
        if self.ppl_cfg is not None:
            self.ppl = True
            self.negative_opt_num = self.ppl_cfg.get('negative_opt_num', 3)
            self.random_seed = self.ppl_cfg.get('random_seed', 0)
            self.strategy = self.ppl_cfg.get('strategy', 'random')
            assert self.strategy in ['random', 'top_p', 'top_similarity'] ,\
                  f'strategy {self.strategy} is not supported'
            random.seed(0)
            self.load_ppl_options()
    
    def load_ppl_options(self):
        print('----generate ppl negative options----')
        self.sentence_list = []
        self.start_index_list = [0] * len(self.data)
        self.end_index_list = [0] * len(self.data) 
        for i, data_item in enumerate(self.data):
            self.start_index_list[i] = len(self.sentence_list)
            self.sentence_list += data_item['sentences']
            self.end_index_list[i] = len(self.sentence_list)


        if self.strategy == 'top_p' or self.strategy == 'top_similarity':
            import numpy as np
            from .utils import Bert_Similarity
            from tqdm import tqdm
            import torch
            bert_similarity = Bert_Similarity(model_path=self.ppl_cfg.get('model_path', 'sentence-transformers/all-MiniLM-L6-v2'))
            bert_embedding = []
            for i in tqdm(range(len(self.sentence_list)), desc="Running bert embedding"):
                bert_embedding.append(bert_similarity.bert_embedding(self.sentence_list[i]))

            embeds = torch.stack(bert_embedding)
            similarity_metric = bert_similarity.embedding_similarity_score(embeds, embeds)
            
            self.candidate_sentence_list = []
            if self.strategy == 'top_p':    
                for i, data_item in tqdm(enumerate(self.data), desc="Running top_p candidates"):
                    sim_metric = similarity_metric[self.start_index_list[i]: self.end_index_list[i]]
                    sim_score = sim_metric.flatten()
                    candidates = self.sentence_list[:]
                    tmp = sim_score.argsort(descending=True).cpu().numpy().tolist()[:250]
                    tmp = [i % len(self.sentence_list) for i in tmp]
                    tmp2 = list(set(tmp))
                    tmp2.sort(key=tmp.index)

                    candidates = np.array(candidates)
                    candidates = candidates[tmp2].tolist()
                    for gt_answer in data_item['sentences']:
                        if gt_answer in candidates:
                            candidates.remove(gt_answer)
                    self.candidate_sentence_list.append(candidates)
                
            else:
                for i, data_item in tqdm(enumerate(self.data), desc="Running top_similarity candidates"):
                    for j in range(self.start_index_list[i], self.end_index_list[i], 1):
                        sim_metric = similarity_metric[j]
                        candidates = self.sentence_list[:]
                        tmp = sim_metric.argsort(descending=True).cpu().numpy().tolist()
                        candidates = np.array(candidates)[tmp].tolist()[:20]
                        for gt_answer in data_item['sentences']:
                            if gt_answer in candidates:
                               candidates.remove(gt_answer)
                        self.candidate_sentence_list.append(candidates)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['id']) if 'id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': os.path.join(self.base_data_path,self.data[index]['image']),
            'gt_answers': self.data[index]['sentences']
        }
        if self.ppl:
            if self.strategy == 'random':
                candidate_sentence_list = self.sentence_list[:]
                for gt_answer in res_dict['gt_answers']:
                    candidate_sentence_list.remove(gt_answer)
                random.shuffle(candidate_sentence_list)
                option_list = candidate_sentence_list[:self.negative_opt_num]
                gt_list = res_dict['gt_answers'][:]
                random.shuffle(gt_list)
                option_list += gt_list[:1]
            elif self.strategy =='top_similarity':
                gt_idx = [i for i in range(5)]
                random.shuffle(gt_idx)
                candidates = self.candidate_sentence_list[self.start_index_list[index] + gt_idx[0]]
                option_list = candidates[:self.negative_opt_num]
                option_list += [res_dict['gt_answers'][gt_idx[0]]]
            elif self.strategy == 'top_p':
                option_list = res_dict['gt_answers'][:]
                option_list += self.candidate_sentence_list[index][:len(res_dict['gt_answers'])]
            res_dict['options'] = option_list
        return res_dict

