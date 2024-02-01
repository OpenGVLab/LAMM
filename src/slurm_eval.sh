
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

parition=$1
model_cfg=$2
recipe_cfg=$3
EXTRA_ARGS=${@:4}

srun -p ${parition} --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python eval.py \
        --model_cfg=${model_cfg} \
        --recipe_cfg=${recipe_cfg} \
        ${EXTRA_ARGS}
