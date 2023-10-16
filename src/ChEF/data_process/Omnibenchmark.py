import json
import os
import tqdm
import math
import random
from shutil import copyfile

class Bamboo_chain:
    def __init__(self, annot_path) -> None:
        annot_data = json.load(open(annot_path,'rb'))
        name2id = {}
        self.id2name = annot_data['id2name']
        self.father2child = annot_data['father2child']
        for key, value in self.id2name.items():
            for name in value:
                name2id[name] = key
        self.name2id = name2id
        self.child2father = annot_data['child2father']
    
    def check_omni_id(self, id):
        if id in self.id2name and (self.id2name[id][0] in self.name2id) and (self.name2id[self.id2name[id][0]] == id):
            return True
        return False

    def get_father(self, id):
        if id not in self.child2father:
            return None
        fathers = self.child2father[id]
        for father in fathers:
            if father not in self.father2child:
                continue
            children = self.father2child[father]
            if id in children:
                return father
        return None

    def get_omni_labels(self, omni_dict, level_num):
        fathers = omni_dict.keys()
        father_dict = {}
        for father in fathers:
            father_dict[father] = []
            for value in omni_dict[father].values():
                value_id = self.name2id[value]
                chain = self.get_label_chain(value_id, level_num - 1)
                if chain is not None:
                    father_dict[father].append(value)
        return father_dict

    
    def get_label_chain(self, label_id, level):
        '''
            chain:['grandgrandfather', 'grandfather', 'father'...]
        '''
        if not self.check_omni_id(label_id):
            return None
        father_id = self.get_father(label_id)
        if father_id is None:
            return None
        if level == 0:
            return [label_id]
        chain = self.get_label_chain(father_id, level-1)
        if chain is not None:
            return [label_id] + chain
        return None
            


def main(source_base_dir, target_base_dir, bamboo_annot_path):
    
    target_img_dir = os.path.join(target_base_dir, 'omnibenchmark_images')
    target_meta_dir = os.path.join(target_base_dir, 'meta_file')
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_meta_dir, exist_ok=True)

    level_num = 4

    
    bamboo_chain = Bamboo_chain(bamboo_annot_path)
    omni_id2name = json.load(open(os.path.join(source_base_dir, 'trainid2name.json'), 'rb'))
    supported_labels = bamboo_chain.get_omni_labels(omni_id2name, level_num)
    omni_data = load_Omni(source_base_dir, 1000, supported_labels)
    for data_item in omni_data:
        label = data_item['label']
        # del data_item['realm_name']
        chain = bamboo_chain.get_label_chain(bamboo_chain.name2id[label], level_num - 1)
        assert chain is not None and len(chain) == level_num
        chain = [bamboo_chain.id2name[chain_item][0] for chain_item in chain]
        chain.reverse()
        data_item['chain'] = chain
    # import ipdb;ipdb.set_trace()

    for data_item in omni_data:
        copyfile(os.path.join(source_base_dir,'data', data_item['realm_name'], 'images', data_item['image'].split('/')[-1]),
                os.path.join(target_img_dir, data_item['image'].split('/')[-1]))
    with open(os.path.join(target_meta_dir, 'Classification_Omnibenchmark.json'), 'w') as f:
        f.write(json.dumps(omni_data, indent=4))

def load_Omni(source_base_dir, subset_len, supported_labels):
    random.seed(0)
    id2name = json.load(open(os.path.join(source_base_dir, 'trainid2name.json'), 'rb'))
    realm_names = id2name.keys()
    assert len(realm_names) == 21
    data = {}
    for realm_name in tqdm.tqdm(realm_names):
        data[realm_name] = []
        annot_path = os.path.join(source_base_dir, 'annotation', realm_name, 'meta', 'test.txt')
        data_list = open(annot_path, 'r', encoding='utf-8').readlines()
        for line_data in data_list:
            image_name = line_data.strip().split(' ')[0]
            label_id = line_data.strip().split(' ')[1]
            label_name = id2name[realm_name][label_id]
            if label_name not in supported_labels[realm_name]:
                continue
            data[realm_name].append(dict(
                image_name = image_name,
                label_id = label_id,
                label_name = label_name,
                realm_name = realm_name
            ))
    realm_len = [len(supported_labels[name]) for name in realm_names]
    sample_len = subset_len // len(realm_len)
    data_for_eval = []
    for realm_name in tqdm.tqdm(realm_names):
        if sample_len == 0 :
            continue
        id_bucket = {}
        label_id2name = {}
        for idx, sample in enumerate(data[realm_name]):
            if sample['label_id'] not in id_bucket:
                id_bucket[sample['label_id']] = []
                label_id2name[sample['label_id']] = sample['label_name']
            id_bucket[sample['label_id']].append(idx)
        value_list = id_bucket.values()
        value_list = [len(value) for value in value_list]
        tmp_total_len = sum(value_list)
        
        value_sample_len = [min(math.ceil(tmp_len * sample_len / tmp_total_len),5) for tmp_len in value_list]


        while sum(value_sample_len) > sample_len:
            shuffle_dict = list(id_bucket.keys())
            random.shuffle(shuffle_dict)
            pop_index = shuffle_dict[0]
            list_index = list(id_bucket.keys()).index(pop_index)
            id_bucket.pop(pop_index)
            value_sample_len.pop(list_index)
            

        print(f'realm_name : {realm_name}, image_num: {sum(value_sample_len)}')
        infostr = ''
        for key, value in zip(id_bucket.keys(), value_sample_len):
            infostr += f'label_name : {label_id2name[key]}, image_num: {value} '
        print(infostr)

        for idx, value in enumerate(id_bucket.values()):
            random.shuffle(value)
            value = value[:value_sample_len[idx]]
            for data_item_idx in value:
                data_item = data[realm_name][data_item_idx]
                data_item_for_eval = dict(
                    image = os.path.join('omnibenchmark_images', data_item['image_name']),
                    src_image = 'omnibenchmark',
                    label = data_item['label_name'],
                    realm_name = data_item['realm_name']
                )
                data_for_eval.append(data_item_for_eval)
    print(f'total_len : {len(data_for_eval)}')
    return data_for_eval

if __name__ == '__main__':
    source_base_dir = '/ssd/home/shizhelun/data/datasets/OmniBenchmark_raw/omnibenchmark_v2_onedrive'
    target_base_dir = '/ssd/home/shizhelun/data/datasets/OmniBenchmark_Bamboo2_v3'
    bamboo_annot_path = '/ssd/home/shizhelun/data/datasets/Bamboo/sensexo_visual_add_academic_add_state_V4.visual.json'
    main(source_base_dir, target_base_dir, bamboo_annot_path)