import os
import sys
import yaml
import argparse
script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(parent_dir)
import json
from metric import build_metric


def load_yaml(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("result_path", type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # config
    base_path = args.result_path
    cfg_path = os.path.join(base_path, 'config.yaml')
    yaml_dict = load_yaml(cfg_path)
    recipe_cfg_path = yaml_dict['recipe']
    recipe_cfg = load_yaml(recipe_cfg_path)

    # dataset
    scenario_cfg = recipe_cfg['scenario_cfg']
    dataset_name = scenario_cfg['dataset_name']

    # eval
    metric_func = build_metric(dataset_name=dataset_name, **recipe_cfg['eval_cfg']['metric_cfg'])

    result_path = None
    if os.path.exists(os.path.join(base_path, 'results.json')):
        with open(os.path.join(base_path, 'results.json'), 'rb') as f:
            json_data = json.load(f)
        result_path = json_data['answer_path']
    else:
        for filename in os.listdir(base_path):
            if filename.endswith('.json'):
                result_path = os.path.join(base_path, filename)
                break
    assert result_path is not None, 'No result file!'
    result = metric_func.metric(result_path)
    with open(os.path.join(base_path, 'results.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict(
            answer_path = result_path,
            result = result
        ), indent=4))

if __name__ == '__main__':
    main()