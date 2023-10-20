PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
cfg_path=configs/evaluation.yaml
srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python tools/eval.py ${cfg_path} --debug
