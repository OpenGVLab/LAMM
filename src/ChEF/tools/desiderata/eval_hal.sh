cfg_path=configs/desiderata_recipes/Hallucination/evaluation.yaml
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/desiderata/eval_hallucination.py ${cfg_path}
