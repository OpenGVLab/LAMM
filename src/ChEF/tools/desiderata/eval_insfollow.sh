cfg_path=configs/desiderata_recipes/Insfollow/evaluation.yaml
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python tools/desiderata/eval_insfollow.py ${cfg_path} 
