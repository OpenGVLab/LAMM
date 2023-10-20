cfg_path=configs/desiderata_recipes/ICL/evaluation.yaml
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python tools/desiderata/eval_icl.py ${cfg_path} --debug