dataset=VOC2012
exp=lamm_13b_lora_186k
base_data_path=data/LAMM-Dataset/2D_Benchmark
token_num=256
layer=-2
answerdir=answers
mkdir -p ${answerdir}/${exp}
results_path=results
mkdir -p ${results_path}/${exp}

python inference.py \
    --model lamm_peft \
    --encoder_pretrain clip \
    --vicuna_ckpt_path ./model_zoo/vicuna_ckpt/13b_v0 \
    --delta_ckpt_path ./model_zoo/lamm_ckpt/${exp}/pytorch_model.pt \
    --max_tgt_len 400 \
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