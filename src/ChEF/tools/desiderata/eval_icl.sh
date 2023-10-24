cfg_path=configs/desiderata_recipes/ICL/evaluation.yaml
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/desiderata/eval_icl.py ${cfg_path}