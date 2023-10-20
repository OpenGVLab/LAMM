from models import get_model
import yaml
import argparse
import os
import numpy as np
import torch
from scenario import dataset_dict
from tools.evaluator import Evaluator
# from evaluation import build as build_evaluater
import datetime

from torch.utils.data import Subset
import json
from utils import rand_acc


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
    dataset_name = scenario_cfg['dataset_name']
    dataset = dataset_dict[dataset_name](**scenario_cfg)
    if args.debug:
        dataset = sample_dataset(dataset, sample_len=16, sample_seed=0)

    # save_cfg
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_base_dir = os.path.join(save_dir, model_cfg['model_name'], dataset_name, time)
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(data=yaml_dict, stream=f, allow_unicode=True)
    print(f'Save results in {save_base_dir}!')

    # evaluate
    if model_cfg['model_name'] in ['MiniGPT-4', 'mPLUG-Owl', 'Otter', 'Kosmos2']:
        recipe_cfg['eval_cfg']['instruction_cfg']['incontext_cfg']['ice_with_image'] = True
    
    ice_nums = [1, 2, 3]
    results = []
    results_path = []
    for ice_num in ice_nums:
        eval_cfg = recipe_cfg['eval_cfg']
        eval_cfg['instruction_cfg']['incontext_cfg']['ice_num'] = ice_num
        evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
        result_path, result = evaluater.evaluate(model)
        results.append(result['vanilla_acc'])
        results_path.append(result_path)
    
    # calculate RIAM
    assert len(results) >1
    acc_icl_average = sum(results[1:])/len(results[1:])
    acc_rand = rand_acc[dataset_name]['vanilla']

    RIAM = (acc_icl_average - results[0]) / (results[0] - acc_rand)
    with open(os.path.join(save_base_dir, 'results.json'), 'w', encoding='utf-8') as f:
        all_results = []
        for i in range(len(results)):
            all_results.append({
                "answer_path": results_path[i],
                "result": results[i]
            })
        all_results.append({"RIAM": RIAM})
        
        json.dump(all_results, f, indent=4)
    

if __name__ == '__main__':
    main()