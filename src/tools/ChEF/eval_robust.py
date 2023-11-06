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
    model_cfg, recipe_cfg, save_dir, sample_len = load_config()

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
    dataset = sample_dataset(dataset, sample_len=sample_len, sample_seed=0)
    # save_cfg
    save_base_dir = os.path.join(base_save_dir, "origin")
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg), stream=f, allow_unicode=True)
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
        dataset = sample_dataset(dataset, sample_len=sample_len, sample_seed=0)
        # save_cfg
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_base_dir = os.path.join(base_save_dir, f'{dataset_name}_{setting[0]}')
        os.makedirs(save_base_dir, exist_ok=True)
        with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg), stream=f, allow_unicode=True)
        print(f'Save {setting[0]} results in {save_base_dir}!')

        # evaluate
        eval_cfg = recipe_cfg['eval_cfg']
        evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
        evaluater.evaluate(model)
    
    #post processing
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