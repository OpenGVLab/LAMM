
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

model_cfg=$1
recipe_cfg=$2

srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python eval.py --model_cfg=${model_cfg} --recipe_cfg=${recipe_cfg}