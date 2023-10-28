import yaml
import os
import datetime
import sys
import json
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
chef_dir = os.path.join(script_dir, '../../../src')
sys.path.append(chef_dir)

from ChEF.evaluator import Evaluator, load_config, sample_dataset
from ChEF.models import get_model
from ChEF.scenario import dataset_dict
from ChEF.scenario.utils import rand_acc

def main():
    model_cfg, recipe_cfg, save_dir, sample_len = load_config()

    # model
    model = get_model(model_cfg)

    # dataset
    scenario_cfg = recipe_cfg['scenario_cfg']
    dataset_name = scenario_cfg['dataset_name']
    dataset = dataset_dict[dataset_name](**scenario_cfg)
    dataset = sample_dataset(dataset, sample_len=sample_len, sample_seed=0)
        
    # save_cfg
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_base_dir = os.path.join(save_dir, model_cfg['model_name'], dataset_name, time)
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg), stream=f, allow_unicode=True)
    print(f'Save results in {save_base_dir}!')

    # evaluate
    if model_cfg['model_name'] in ['MiniGPT-4', 'mPLUG-Owl', 'Otter', 'Kosmos2']:
        recipe_cfg['eval_cfg']['instruction_cfg']['incontext_cfg']['ice_with_image'] = True
    
    ice_nums = [0, 1, 2, 3]
    results = []
    results_path = []
    for ice_num in ice_nums:
        eval_cfg = recipe_cfg['eval_cfg']
        eval_cfg['instruction_cfg']['incontext_cfg']['ice_num'] = ice_num
        evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
        result_path, result = evaluater.evaluate(model)
        if 'vanilla_acc' in result:
            results.append(result['vanilla_acc'])
        else:
            results.append(result['ACC'])
        results_path.append(result_path)
    
    # calculate RIAM
    assert len(results) >1
    acc_icl_average = sum(results[1:])/len(results[1:])
    acc_rand = rand_acc[dataset_name]['vanilla']
    RIAM = ((acc_icl_average - results[0]) / (results[0] - acc_rand)) * 50 + 50
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