partition=$1
exp=$2
dataset=$3

base_data_path=../data/2D_Benchmark
token_num=256
layer=-2
answerdir=../answers
mkdir -p ${answerdir}/${exp}
results_path=../results
mkdir -p ${results_path}/${exp}

srun -p ${partition} --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python inference_2d.py \
        --model octavius \
        --encoder_pretrain clip \
        --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0 \
        --delta_ckpt_path ../ckpt/${exp}/pytorch_model.pt \
        --max_tgt_len 400 \
        --peft_type moe_lora \
        --moe_lora_num_experts 4 \
        --moe_gate_mode top2_gate \
        --octavius_modality image \
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --num_vision_token ${token_num} \
        --vision_output_layer ${layer} \
        --conv_mode simple \
        --dataset-name ${dataset} \
        --base-data-path ${base_data_path} \
        --inference-mode common \
        --bs 32 \
        --answers-dir ${answerdir}/${exp} \
