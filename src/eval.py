import argparse
import datetime
import numpy as np
import os
import torch
from torch.utils.data import Subset
import yaml

from ChEF.models import get_model
from ChEF.scenario import dataset_dict
from ChEF.tools.evaluator import Evaluator


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
    parser.add_argument("--model_cfg", type=str, required=True)
    parser.add_argument("--recipe_cfg", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="../results")

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


def main():
    args = parse_args()
    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    # config
    model_cfg = load_yaml(args.model_cfg)
    recipe_cfg = load_yaml(args.recipe_cfg)
    save_dir = args.save_dir

    # model
    model = get_model(model_cfg)

    # dataset
    scenario_cfg = recipe_cfg['scenario_cfg']
    dataset_name = scenario_cfg['dataset_name']
    if scenario_cfg['dataset_name'] == "OctaviusPCLDataset":
        # delete dataset_name from scenario_cfg
        scenario_cfg.pop('dataset_name')
    dataset = dataset_dict[dataset_name](**scenario_cfg)
    if args.debug:
        dataset = sample_dataset(dataset, sample_len=16, sample_seed=0)
    if args.sample_len != -1:
        dataset = sample_dataset(dataset, sample_len=args.sample_len, sample_seed=0)

    # save_cfg
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_base_dir = os.path.join(save_dir, model_cfg['model_name'], dataset_name, time)
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg), stream=f, allow_unicode=True)
    print(f'Save results in {save_base_dir}!')

    # evaluate
    eval_cfg = recipe_cfg['eval_cfg']
    evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
    evaluater.evaluate(model)

if __name__ == '__main__':
    main()
