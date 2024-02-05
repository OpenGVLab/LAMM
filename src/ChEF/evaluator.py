import os
import json
import yaml
import shutil
import argparse
import numpy as np
import datetime
from torch.utils.data import Subset
from time import sleep
from .instruction import build_instructionhandler
from .inferencer import build_inferencer
from .metric import build_metric

class Evaluator:
    def __init__(
        self,
        dataset,
        save_base_dir,
        cfg,
        dist_args,
        **kwargs) -> None:
        self.dataset = dataset
        self.dataset_name = self.dataset.dataset_name
        self.task_name = self.dataset.task_name
        self.save_base_dir = save_base_dir
        instruction_cfg = cfg['instruction_cfg']
        self.instruction_handler = build_instructionhandler(
            task_name=self.task_name, 
            dataset=self.dataset, 
            **instruction_cfg)
        inferencer_cfg = cfg['inferencer_cfg']
        self.inferencer = build_inferencer(
            dataset_name=self.dataset_name,
            save_base_dir=save_base_dir,
            instruction_handler=self.instruction_handler,
            dist_args=dist_args,
            **inferencer_cfg)
        
        metric_cfg = cfg['metric_cfg']
        self.metric = build_metric(
            dataset_name=self.dataset_name, 
            **metric_cfg)
        self.dist_args = dist_args

    def check_all_rank_done(self):
        for i in range(self.dist_args['world_size']):
            if not os.path.exists(os.path.join(self.save_base_dir, \
                'tmp', f'{self.dataset_name}_{i}.json')):
                return False
        return True
        
    def merge_result_data(self):
        while True:
            sleep(1.0)
            if self.check_all_rank_done():
                all_results = []
                for i in range(self.dist_args['world_size']):
                    with open(os.path.join(self.save_base_dir, \
                        'tmp', f'{self.dataset_name}_{i}.json'), 'rb') as f:
                        rank_result = json.load(f)
                        all_results += rank_result
                time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                answer_path = os.path.join(self.save_base_dir, f"{self.dataset_name}_{time}.json")
                with open(answer_path, "w") as f:
                    f.write(json.dumps(all_results, indent=4))
                shutil.rmtree(os.path.join(self.save_base_dir, 'tmp'))
                return answer_path

    def evaluate(self, model):
        self.inferencer.inference(model, self.dataset)

        if self.dist_args['world_size'] > 1 and self.dist_args['global_rank'] != 0:
            return
        if self.dist_args['world_size'] > 1:
            results_path = self.merge_result_data()
        else:
            results_path = self.inferencer.results_path

        result = self.metric.metric(results_path)
        with open(os.path.join(self.save_base_dir, 'results.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(dict(
                answer_path = results_path,
                result = result
            ), indent=4))


class CustomSubset(Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.task_name = dataset.task_name
        self.dataset_name = dataset.dataset_name
        self.data = dataset.data
        if hasattr(dataset, 'system_msg'):
            self.system_msg = dataset.system_msg


def load_yaml(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model_cfg", type=str, required=True)
    parser.add_argument("--recipe_cfg", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="../results")
    parser.add_argument("--time", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sample_len", type=int, default=-1)
    args = parser.parse_args()
    return args

def sample_dataset(dataset, sample_len=1000, sample_seed=0, dist_args=None):
    if sample_len == -1:
        pass
    elif len(dataset) > sample_len:
        np.random.seed(sample_seed)
        random_indices = np.random.choice(
            len(dataset), sample_len, replace=False
        )
        dataset = CustomSubset(dataset, random_indices)
    if dist_args is not None and dist_args['world_size'] > 1:
        # split dataset
        rank = dist_args['global_rank']
        world_size = dist_args['world_size']
        data_per_rank = len(dataset) // world_size
        start = rank * data_per_rank
        end = (rank + 1) * data_per_rank
        if rank == world_size - 1:
            end = len(dataset)
        if isinstance(dataset, CustomSubset):
            original_indices = dataset.indices
            sliced_indices = original_indices[start:end]
            dataset = CustomSubset(dataset.dataset, sliced_indices)
        else:
            sliced_indices = [i for i in range(len(dataset))][start:end]
            dataset = CustomSubset(dataset, sliced_indices)
    return dataset

def load_config():
    args = parse_args()
    # if args.device != -1:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    # config
    model_cfg = load_yaml(args.model_cfg)
    recipe_cfg = load_yaml(args.recipe_cfg)
    save_dir = args.save_dir
    if args.debug:
        sample_len = 32
    elif args.sample_len != -1:
        sample_len = args.sample_len
    else:
        sample_len = -1
    return model_cfg, recipe_cfg, save_dir, sample_len, args.time