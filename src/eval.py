import os
import yaml
import torch
from dist import init_distributed_mode

from ChEF.models import get_model
from ChEF.scenario import dataset_dict
from ChEF.evaluator import Evaluator, load_config, sample_dataset

def get_useable_cuda():
    test_tensor = torch.zeros((1))
    useable_cuda = []
    for i in range(8):
        try:
            test_tensor.to(device=i)
            useable_cuda.append(i)
        except:
            continue
    return useable_cuda

def main(dist_args):
    model_cfg, recipe_cfg, save_dir, sample_len, time = load_config()

    # model
    devices = get_useable_cuda()
    model = get_model(model_cfg, device=devices[dist_args['global_rank']])
    # dataset
    scenario_cfg = recipe_cfg['scenario_cfg']
    dataset_name = scenario_cfg['dataset_name']
    dataset = dataset_dict[dataset_name](**scenario_cfg)
    # sample dataset
    dataset = sample_dataset(dataset, sample_len=sample_len, sample_seed=0, dist_args=dist_args)

    # save_cfg
    save_base_dir = os.path.join(save_dir, model_cfg['model_name'], dataset_name, time)
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg), stream=f, allow_unicode=True)
    print(f'Save results in {save_base_dir}!')

    # evaluate
    eval_cfg = recipe_cfg['eval_cfg']
    evaluater = Evaluator(dataset, save_base_dir, eval_cfg, dist_args=dist_args)
    evaluater.evaluate(model)

if __name__ == '__main__':
    dist_args = init_distributed_mode()
    main(dist_args)