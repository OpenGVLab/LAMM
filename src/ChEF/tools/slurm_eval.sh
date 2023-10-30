PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# cfg_path=/mnt/petrelfs/chenzeren/LAMM/src/config/ChEF/eval.yaml

model_cfg=/mnt/petrelfs/chenzeren/LAMM/src/config/ChEF/models/octavius_2d+3d.yaml
recipe_cfg=/mnt/petrelfs/chenzeren/LAMM/src/config/ChEF/scenario_recipes/LAMM/VOC2012.yaml

srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python /mnt/petrelfs/chenzeren/LAMM/src/eval.py --model_cfg=${model_cfg} --recipe_cfg=${recipe_cfg}