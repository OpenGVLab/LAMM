from torch.utils.data import Subset
import yaml
import numpy as np
import argparse
import os
import json
from .instruction import build_instructionhandler
from .inferencer import build_inferencer
from .metric import build_metric


class Evaluator:
    def __init__(self,
                 dataset,
                 save_base_dir,
                 cfg,
                 **kwargs) -> None:
        self.dataset = dataset
        self.dataset_name = self.dataset.dataset_name
        self.task_name = self.dataset.task_name
        self.save_base_dir = save_base_dir
        instruction_cfg = cfg['instruction_cfg']
        self.instruction_handler = build_instructionhandler(task_name=self.task_name, dataset=self.dataset, **instruction_cfg)
        inferencer_cfg = cfg['inferencer_cfg']
        self.inferencer = build_inferencer(dataset_name = self.dataset_name,
                                           save_base_dir = save_base_dir,
                                           instruction_handler = self.instruction_handler,
                                           **inferencer_cfg)
        
        metric_cfg = cfg['metric_cfg']
        self.metric = build_metric(dataset_name=self.dataset_name, 
                                   **metric_cfg)
        
    def evaluate(self, model):
        model.ice_imgs_emb = None
        self.inferencer.inference(model, self.dataset)
        results_path = self.inferencer.results_path
        result = self.metric.metric(results_path)
        with open(os.path.join(self.save_base_dir, 'results.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(dict(
                        answer_path = results_path,
                        result = result
                    ), indent=4))
    
        return results_path, result


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

def load_config():
    args = parse_args()
    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    # config
    model_cfg = load_yaml(args.model_cfg)
    recipe_cfg = load_yaml(args.recipe_cfg)
    save_dir = args.save_dir
    if args.debug:
        sample_len = 16
    elif args.sample_len != -1:
        sample_len = args.sample_len
    else:
        sample_len = -1
    return model_cfg, recipe_cfg, save_dir, sample_len

def build_evaluator(dataset, task_name, save_base_dir, eval_cfg, **kwargs):
    return Evaluator(dataset=dataset, save_base_dir=save_base_dir, cfg=eval_cfg)
