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
from tools.desiderata.utils import rand_acc

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
    parser.add_argument("--sample_len", type=int, default=-1)
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


def compute_RRM(origin_acc, crp_acc, dataset_name):
    rd_acc = rand_acc[dataset_name]['vanilla']
    return (crp_acc - rd_acc) / (origin_acc - rd_acc) * 100
    
def get_acc(res_dict):
    acc_keys = ['ACC','vanilla_acc']
    for acc_key in acc_keys:
            if acc_key in res_dict:
                origin_acc = res_dict[acc_key]
                return origin_acc
    return 0
    

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
    
    settings = [('ic', True, False), ('tc', False, True), ('ictc', True, True)]
    dataset_name = scenario_cfg['dataset_name']
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    base_save_dir = os.path.join(save_dir, model_cfg['model_name'],'Robust', dataset_name, time)
    # origin 
    scenario_cfg['img_crp'], scenario_cfg['text_crp'] = False, False
    dataset = dataset_dict[dataset_name](**scenario_cfg)
    if args.debug:
        dataset = sample_dataset(dataset, sample_len=16, sample_seed=0)
    if args.sample_len != -1:
        dataset = sample_dataset(dataset, sample_len=args.sample_len, sample_seed=0)
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
        scenario_cfg['img_crp'], scenario_cfg['text_crp'] = setting[1], setting[2]
        dataset_name = scenario_cfg['dataset_name']
        dataset = dataset_dict[dataset_name](**scenario_cfg)
        if args.debug:
            dataset = sample_dataset(dataset, sample_len=16, sample_seed=0)
        if args.sample_len != -1:
            dataset = sample_dataset(dataset, sample_len=args.sample_len, sample_seed=0)
        # save_cfg
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_base_dir = os.path.join(base_save_dir, f'{dataset_name}_{setting[0]}')
        os.makedirs(save_base_dir, exist_ok=True)
        with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(data=yaml_dict, stream=f, allow_unicode=True)
        print(f'Save {setting[0]} results in {save_base_dir}!')

        # evaluate
        eval_cfg = recipe_cfg['eval_cfg']
        evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
        evaluater.evaluate(model)
    
    #post processing

    acc_keys = ['ACC','vanilla_acc']
    with open(os.path.join(origin_save_base_dir, 'results.json'),'r') as f:
        res_dict = json.load(f)['result']
        origin_acc = get_acc(res_dict)

    final_res = [{'img_crp': False, 
               'text_crp': False, 
               'dir': origin_save_base_dir, 
               'Acc': origin_acc, 
               'RRM': 100,
               'RR': 100,
               }]
    for setting in settings:
        dir = os.path.join(base_save_dir, f'{dataset_name}_{setting[0]}')
        acc_json_path = os.path.join(dir, 'results.json')
        with open(acc_json_path,'r') as f:
            acc_data = json.load(f)
        res_dict = acc_data['result']
        acc = get_acc(res_dict)
        rrm = compute_RRM(origin_acc, acc, dataset_name)
        res = {'img_crp': setting[1], 
               'text_crp': setting[2], 
               'dir': dir, 
               'Acc': acc, 
               'RRM': rrm,
               'RR': acc / origin_acc * 100,
               }
    
        final_res.append(res)
    
    for res in final_res:
        print(res)
        
    with open(os.path.join(base_save_dir, 'Robust_Results.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_res, indent=4))


if __name__ == '__main__':
    main()