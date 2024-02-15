import os
import json
from torch.utils.data import Dataset
import random
import inflect
inflect_engine = inflect.engine()

class CIFAR10Dataset(Dataset):
    task_name = 'coarse_grained_classification'
    dataset_name = 'CIFAR10'
    cifar_classes = ['cat','ship','airplane','frog','automobile','truck', 'dog', 'horse', 'deer', 'bird']
    def __init__(self, base_data_path, ppl=False, split='', **kwargs):
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,'meta_file', f'Classification_{split}CIFAR10.json')
        self.split = split.replace('_','')
        self.data = json.load(open(json_path,'rb'))
        self.ppl = ppl
        # for i in range(len(self.cifar_classes)):
        #     self.cifar_classes[i] = inflect_engine.a(self.cifar_classes[i])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['id']) if 'id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': os.path.join(self.base_data_path,self.data[index]['image']),
            'gt_answers': self.data[index]['label'],
        }
        if self.ppl:
            res_dict['options'] = self.cifar_classes
        return res_dict


class OmnibenchmarkDataset(Dataset):
    '''
        Omnibenchmark is a fine-grained classification dataset. The default setting is multiturn ppl.
    '''
    task_name = 'fine_grained_classification'
    dataset_name = 'Omnibenchmark'
    def __init__(self, 
                 bamboo_tree_path, 
                 base_data_path, 
                 multi_turn=True,
                 ppl_cfg = {
                    'negative_opt_num': 3,
                    'random_seed': 0,
                 },
                 **kwargs):
        self.bamboo_tree_path = bamboo_tree_path
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,'meta_file', f'Classification_Omnibenchmark.json')
        self.data = json.load(open(json_path,'rb'))
        self.ppl_cfg = ppl_cfg
        self.multi_turn = multi_turn
        if self.ppl_cfg is None:
            return 
        self.negative_opt_num = self.ppl_cfg.get('negative_opt_num', 3)
        self.random_seed = self.ppl_cfg.get('random_seed', 0)
        random.seed(self.random_seed)
        self.load_ppl_options()
            

    def load_ppl_options(self):
        def check(id):
            if id in self.id2name and (self.id2name[id][0] in self.name2id) and (self.name2id[self.id2name[id][0]] == id):
                return True
            return False
        print('----generate ppl negative options----')
        self.disjointset = []
        self.name2djsid = dict()
        
        # load bamboo
        annot_data = json.load(open(self.bamboo_tree_path,'rb'))
        self.id2name = annot_data['id2name']
        self.father2child = annot_data['father2child']
        name2id = {}
        for key, value in self.id2name.items():
            for name in value:
                name2id[name] = key
        self.name2id = name2id
        self.child2father = annot_data['child2father']
        
        djsid = -1
        for data_item in self.data:
            data_chain = data_item['chain']
            for i, label_name in enumerate(data_chain):
                if label_name in self.name2djsid:
                    continue
                djsid += 1
                tmp = [label_name]
                self.name2djsid[label_name] = djsid
                if i == 0:
                    children = self.father2child[self.child2father[self.name2id[label_name]][0]]
                else:
                    children = self.father2child[self.name2id[data_chain[i-1]]]
                for child in children:
                    if not check(child):
                        continue
                    if self.name2id[label_name] == child:
                        continue
                    child_name = self.id2name[child][0]
                    tmp.append(child_name)
                    self.name2djsid[child_name] = djsid
                
                assert label_name in tmp
                self.disjointset.append(tmp)    
        self.ppl_options = []
        for index in range(len(self.data)):
            ppl_options = []
            for label_name in self.data[index]['chain']:
                label_option_set = self.disjointset[self.name2djsid[label_name]][:]
                label_option_set.remove(label_name)
                random.shuffle(label_option_set)
                label_option_list = label_option_set[:self.negative_opt_num]
                label_option_list += [label_name]
                random.shuffle(label_option_list)
                ppl_options.append(label_option_list)
            self.ppl_options.append(ppl_options)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['id']) if 'id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': os.path.join(self.base_data_path,self.data[index]['image']),
            'gt_answers': self.data[index]['chain'],
            'realm_name' : self.data[index]['realm_name'],
        }
        ppl_options = self.ppl_options[index]
        if self.multi_turn:
            # multiturn direct inference
            multi_turn_prefix = [dict(
                prompt_idx = 0,
                prefix = None
            )]
            for i in range(1, len(ppl_options), 1):
                multi_turn_prefix.append(dict(
                    prompt_idx = 1,
                    prefix = self.data[index]['chain'][i-1]
                ))
            res_dict['multi_turn_prefix'] = multi_turn_prefix

        if self.ppl_cfg is not None:
            if self.multi_turn:
                res_options = [dict(
                    prompt_idx = 0,
                    prefix = None,
                    options = ppl_options[0]
                )]
                for i in range(1,len(ppl_options),1):
                    res_options.append(dict(
                        prompt_idx = 1,
                        prefix = self.data[index]['chain'][i-1],
                        options = ppl_options[i]
                    ))
                res_dict['options'] = res_options
            else:
                res_dict['options'] = ppl_options[-1]
        return res_dict


if __name__ == '__main__':
    # omnidata = OmnibenchmarkDataset(
    #     bamboo_tree_path='../../../data/Bamboo/sensexo_visual_add_academic_add_state_V4.visual.json', 
    #     base_data_path='../../../data/ChEF/OmniBenchmark_Bamboo',
    #     multi_turn=False,
    #     ppl_cfg={})
    cifardata = CIFAR10Dataset(
        '../../../data/LAMM/2D_Benchmark'
    )
    import ipdb;ipdb.set_trace()