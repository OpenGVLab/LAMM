cfg_path=configs/desiderata_recipes/Robust/evaluation.yaml
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/desiderata/eval_robust.py ${cfg_path}
