yaml_dict=(ScanNet ScanRefer ScanQA)

for dataset in ${yaml_dict[*]}; do
    
    python eval.py --model_cfg config/ChEF/models/lamm_3d.yaml --recipe_cfg config/ChEF/scenario_recipes/LAMM/${dataset}.yaml

done