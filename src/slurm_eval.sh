START_TIME=`date +%Y%m%d-%H:%M:%S`

parition=$1
GPUS=$2
model_cfg=$3
recipe_cfg=$4
EXTRA_ARGS=${@:5}
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

LOG_FILE=../logs/evaluation_${START_TIME}.log

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    srun -p ${parition} -J ChEF_eval --gres=gpu:${GPUS_PER_NODE} --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} --kill-on-bad-exit \
            python eval.py \
                --time ${START_TIME} \
                --model_cfg=${model_cfg} \
                --recipe_cfg=${recipe_cfg} \
                ${EXTRA_ARGS} \
                2>&1 | tee -a $LOG_FILE > /dev/null &

    sleep 0.5s;
    tail -f ${LOG_FILE}