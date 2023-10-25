import os
import json
from torch.utils.data import Dataset
import random
from .lamm_sysmsg import common_task2sysmsg


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


class CIFAR10LAMMDataset(Dataset):
    task_name = 'classification_lamm'
    dataset_name = 'CIFAR10'
    CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    def __init__(self, base_data_path, **kwargs):
        super().__init__()
        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Classification_CIFAR10.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['Classification']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        random_labels = random.sample(self.CIFAR10_LABELS, len(self.CIFAR10_LABELS))
        shuffled_labels_str = ", ".join(random_labels)

        additional_sentence = "Please choose a label from the following shuffled categories: "

        data_dict = {
            'id' : data_id,
            'label' : item['label'],
            'question' : f"{item['query']} {additional_sentence}({shuffled_labels_str})",
            'image_path' : os.path.join(self.base_data_path, item['image']),
        }
        return data_dict


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


class CelebAHairDataset(Dataset):
    task_name = 'Facial_cls_lamm'
    dataset_name = 'CelebA(Hair)'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Facial_Classification_CelebA(Hair).json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['Facial_Classification']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)
 
        data_dict = {
            'id' : data_id,
            'image_path' : os.path.join(self.base_data_path, item['image']),
            'question' : item['query'],
            'gt' : item['attr'],
        }
        return data_dict


class CelebASmileDataset(Dataset):
    task_name = 'Facial_cls_lamm'
    dataset_name = 'CelebA(Smile)'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Facial_Classification_CelebA(Smile).json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['Facial_Classification']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)
 
        data_dict = {
            'id' : data_id,
            'image_path' : os.path.join(self.base_data_path, item['image']),
            'question' : item['query'],
            'gt' : item['attr'],
        }
        return data_dict