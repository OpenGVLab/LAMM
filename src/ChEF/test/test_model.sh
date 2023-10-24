model_cfg=$1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python test/test_model.py ${model_cfg} --debug