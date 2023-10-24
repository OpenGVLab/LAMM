cfg_path=configs/evaluation.yaml
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/eval.py ${cfg_path}
