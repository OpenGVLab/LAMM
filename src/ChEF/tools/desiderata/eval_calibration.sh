cfg_path=configs/desiderata_recipes/Calibration/evaluation.yaml
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/desiderata/eval_calibration.py ${cfg_path} 