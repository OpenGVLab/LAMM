yaml_dict=(ScanNet ScanRefer ScanQA)

for dataset in ${yaml_dict[*]}; do
    
    srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python eval.py --model_cfg config/ChEF/models/lamm_3d.yaml --recipe_cfg config/ChEF/scenario_recipes/LAMM/${dataset}.yaml

done