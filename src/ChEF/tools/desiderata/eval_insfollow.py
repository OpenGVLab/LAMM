from models import get_model
import yaml
import argparse
import os
import numpy as np
import torch
import json
from scenario import dataset_dict
from tools.evaluator import Evaluator
# from evaluation import build as build_evaluater
import datetime

from torch.utils.data import Subset
class CustomSubset(Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.task_name = dataset.task_name
        self.dataset_name = dataset.dataset_name
        self.data = dataset.data


def load_yaml(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("cfg_path", type=str)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args

def sample_dataset(dataset, sample_len=1000, sample_seed=0):
    if sample_len == -1:
        return dataset
    if len(dataset) > sample_len:
        np.random.seed(sample_seed)
        random_indices = np.random.choice(
            len(dataset), sample_len, replace=False
        )
        dataset = CustomSubset(dataset, random_indices)
    return dataset

def find_res_json(directory, dataset_name):
    scienceqa_files = []
    if not os.path.isdir(directory):
        return directory

    #import ipdb;ipdb.set_trace()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(dataset_name) and file.endswith(".json"):
                file_path = os.path.join(root, file)
                scienceqa_files.append(file_path)
    return scienceqa_files[0]

def extract_result_json(subdirs):
    result_json_list = []
    config_list = []
    acc_list = []
    cnt=0
    for subdir in subdirs:
        result_json_path = os.path.join(subdir, 'results.json')
        cfg_path = os.path.join(subdir, 'config.yaml')
        if cnt>=ct:
            return result_json_list,config_list,acc_list
        if os.path.exists(result_json_path):
            with open(result_json_path, 'r') as json_file:
                acc_data=json.load(json_file)
                acc_list.append(acc_data[0]['result'])
            result_json_path=find_res_json(subdir)
            with open(result_json_path, 'r') as json_file:
                result_data = json.load(json_file)
                result_json_list.append(result_data)
            config_list.append(load_yaml(cfg_path))
            cnt+=1

    return result_json_list, config_list, acc_list

def compute_MR(origin, target):
    mct, total = 0, 0
    res={}
    for o,t in zip(origin, target):
        main_idx = str(int(o['id']) % int(1e6))
        if main_idx in res:
            continue
        oid = o['options'].index(o['answer'])
        res[main_idx]=1
        #hh+=1.0/len(o['gt_choices'])
        total+=1
        if not (o['id']==t['id']):
            import ipdb;ipdb.set_trace()
        assert o['id']==t['id']
        try:
            tid = t['options'].index(t['answer'])
        except:
            import ipdb;ipdb.set_trace()
        if oid==tid:
            mct+=1
    #print(f"mct:{mct}, cct:{cct}")#follow original & stick to original for unnatural
    #print(total)
    return mct/total*100

def main():
    args = parse_args()
    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    # config
    yaml_dict = load_yaml(args.cfg_path)
    model_cfg_path = yaml_dict['model']
    save_dir = yaml_dict['save_dir']
    recipe_cfg_path = yaml_dict['recipe']
    model_cfg = load_yaml(model_cfg_path)
    recipe_cfg = load_yaml(recipe_cfg_path)

    # model
    model = get_model(model_cfg)
    
    # dataset
    scenario_cfg = recipe_cfg['scenario_cfg']
    settings = [('natural',0), ('natural',1), ('natural',2), ('neutral',0), ('neutral',1) ,('unnatural',0)]
    dataset_name = scenario_cfg['dataset_name']
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    base_save_dir = os.path.join(save_dir, model_cfg['model_name'],'Instruct_follow', dataset_name, time)
    # origin 
    scenario_cfg['option_map'] = None
    dataset = dataset_dict[dataset_name](**scenario_cfg)
    if args.debug:
        dataset = sample_dataset(dataset, sample_len=16, sample_seed=0)
    # save_cfg
    save_base_dir = os.path.join(base_save_dir, "origin")
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(data=yaml_dict, stream=f, allow_unicode=True)
    print(f'Save origin results in {save_base_dir}!')
    # evaluate
    eval_cfg = recipe_cfg['eval_cfg']
    evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
    evaluater.evaluate(model)
    origin_save_base_dir = save_base_dir
    for setting in settings:
        ins_dict={
                'type':setting[0],
                'ids' : setting[1]
                }
        scenario_cfg['option_map'] = ins_dict
        dataset = dataset_dict[dataset_name](**scenario_cfg)
        if args.debug:
            dataset = sample_dataset(dataset, sample_len=16, sample_seed=0)
        # save_cfg
        save_base_dir = os.path.join(base_save_dir, f"{setting[0]}_{setting[1]}")
        os.makedirs(save_base_dir, exist_ok=True)

        with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(data=yaml_dict, stream=f, allow_unicode=True)
        print(f'Save {setting[0]}_{setting[1]} results in {save_base_dir}!')

        # evaluate
        eval_cfg = recipe_cfg['eval_cfg']
        evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
        evaluater.evaluate(model)

    #post processing
    with open(find_res_json(origin_save_base_dir, dataset_name),'r') as f:
        origin_res = json.load(f)

    types_dirs = {'natural':[],'neutral':[],'unnatural':[]}
    types_accs = {'natural':[],'neutral':[],'unnatural':[]}
    types_mrs = {'natural':[],'neutral':[],'unnatural':[]}
    for setting in settings:
        dir = os.path.join(base_save_dir, f"{setting[0]}_{setting[1]}")
        types_dirs[setting[0]].append(dir)
        acc_json_path = os.path.join(dir, 'results.json')
        with open(acc_json_path,'r') as f:
            acc_data = json.load(f)
        types_accs[setting[0]].append(acc_data['result'])
        result_json_path = find_res_json(dir, dataset_name)
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)
        mr = compute_MR(origin_res, result_data)
        types_mrs[setting[0]].append(mr)
        print(f"{setting[0]}_{setting[1]}: Acc: {acc_data['result']}, follow_MR: {mr}")

    avg_acc, avg_mr = 0, 0
    for type,accs in types_accs.items():
        tmp = 0
        for acc in accs:
            tmp+=acc
        avg_acc+=tmp/len(accs)
    avg_acc/=3
    for type,mrs in types_mrs.items():
        tmp = 0
        for mr in mrs:
            tmp+=mr
        avg_mr+=tmp/len(mrs)
    avg_mr/=3
    
    print(f'weighted_avg_MR: {avg_mr}, weighted_avg_Acc: {avg_acc}')
    final_res = {
    'res_dirs': types_dirs,
    'Accs': types_accs,
    'MRs': types_mrs,
    'weighted_avg_MR': avg_mr,
    'weighted_avg_Acc':avg_acc,
    }
    
    with open(os.path.join(base_save_dir, 'Instruction_Follow_Results.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_res, indent=4))

    

if __name__ == '__main__':
    main()