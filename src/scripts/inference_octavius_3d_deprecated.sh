partition=$1
exp=$2

token_num=256
layer=-2
visfeat_type=local
dataset='scannet'
task='Caption'

base_data_path=../data/3D_Benchmark
vision_root_path=/mnt/petrelfs/chenzeren/LAMM-Det3d/data/3D_Instruct

answerdir=../answers
mkdir -p ${answerdir}/${exp}
results_path=../results
mkdir -p ${results_path}/${exp}


srun -p ${partition} --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python inference_3d.py \
    --model octavius \
    --encoder_pretrain clip \
    --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0 \
    --delta_ckpt_path ../ckpt/${exp}/pytorch_model.pt \
    --max_tgt_len 800 \
    --peft_type moe_lora \
    --moe_lora_num_experts 6 \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --vision_feature_type ${visfeat_type} \
    --num_vision_token ${token_num} \
    --vision_output_layer ${layer} \
    --conv_mode simple \
    --dataset-name ${dataset} \
    --task-name ${task} \
    --base-data-path ${base_data_path} \
    --vision-root-path ${vision_root_path} \
    --inference-mode common \
    --bs 1 \
    --answers-dir ${answerdir}/${exp} \
