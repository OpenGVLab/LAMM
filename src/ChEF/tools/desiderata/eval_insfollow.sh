cfg_path=configs/desiderata_recipes/Insfollow/evaluation.yaml
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/desiderata/eval_insfollow.py ${cfg_path} 
