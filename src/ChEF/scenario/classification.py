import os
import json
from torch.utils.data import Dataset
import random
class CIFAR10Dataset(Dataset):
    task_name = 'coarse_grained_classification'
    dataset_name = 'CIFAR10'
    def __init__(self, base_data_path, ppl=False, split='', **kwargs):
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,'meta_file', f'Classification_{split}CIFAR10.json')
        self.split = split.replace('_','')
        self.data = json.load(open(json_path,'rb'))
        self.ppl = ppl
    
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
            res_dict['options']=['cat','ship','airplane','frog','automobile','truck', 'dog', 'horse', 'deer', 'bird']
        return res_dict

class OmnibenchmarkDataset(Dataset):
    task_name = 'fine_grained_classification'
    dataset_name = 'Omnibenchmark'
    def __init__(self, 
                 bamboo_tree_path, 
                 base_data_path, 
                 ppl_cfg = None,
                 **kwargs):
        self.bamboo_tree_path = bamboo_tree_path
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,'meta_file', f'Classification_Omnibenchmark.json')
        self.data = json.load(open(json_path,'rb'))
        self.ppl_cfg = ppl_cfg
        self.ppl = False
        if self.ppl_cfg is not None:
            self.ppl = True
            self.negative_opt_num = self.ppl_cfg.get('negative_opt_num', 3)
            self.random_seed = self.ppl_cfg.get('random_seed', 0)
            self.single_turn = self.ppl_cfg.get('single_turn', True)
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
        if self.ppl:
            ppl_options = self.ppl_options[index]
            if self.single_turn:
                options = []
                for ppl_option in ppl_options:
                    options += ppl_option
                random.shuffle(options)
                res_dict['options'] = options
            else:
                res_options = [dict(
                    fore_label = None,
                    options = ppl_options[0]
                )]
                for i in range(1,len(ppl_options),1):
                    res_options.append(dict(
                        fore_label = self.data[index]['chain'][i-1],
                        options = ppl_options[i]
                    ))
                res_dict['options'] = res_options
        return res_dict

if __name__ == '__main__':
    dataset = OmnibenchmarkDataset(bamboo_tree_path = '/cpfs01/user/shizhelun/shizhelun/data/dataset/Bamboo/sensexo_visual_add_academic_add_state_V4.visual.json', 
                 base_data_path = '/cpfs01/user/shizhelun/shizhelun/data/dataset/OmniBenchmark_Bamboo2_v3',ppl_cfg=dict(single_turn=False))
    data = dataset[0]
    import ipdb;ipdb.set_trace()