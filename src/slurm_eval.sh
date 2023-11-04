
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

model_cfg=config/ChEF/models/octavius_2d+3d.yaml
recipe_cfg=config/ChEF/scenario_recipes/Octavius3D/scan_caption_direct3d.yaml

srun -p <YOUR_PARTITION> --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python eval.py --model_cfg=${model_cfg} --recipe_cfg=${recipe_cfg}