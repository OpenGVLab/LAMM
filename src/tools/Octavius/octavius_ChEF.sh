# choose one model_cfg from 2d, 3d, 2d+3d
model_cfg=config/ChEF/models/octavius_2d.yaml
recipe_cfg_list=(CIFAR10)

for dataset in ${recipe_cfg_list[*]}; do
    srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
        python eval.py \
            --model_cfg ${model_cfg} \
            --recipe_cfg config/ChEF/scenario_recipes/LAMM/${dataset}.yaml
done