cfg_path=ChEF/configs/lamm_eval/evaluation_cap_nr3d.yaml

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python ChEF/tools/eval.py ${cfg_path}
