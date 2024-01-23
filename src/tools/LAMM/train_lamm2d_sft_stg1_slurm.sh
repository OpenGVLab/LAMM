#!/bin/bash
numgpu=8

partition=$1
exp=$2
visfeat_type=local

now=$(date +"%Y%m%d_%H%M%S")
ckpt_dir=../ckpt
mkdir -p ${ckpt_dir}/${exp}/log_rest/

srun -p ${partition} -J ${exp} --gres=gpu:${numgpu} --ntasks-per-node 1 --kill-on-bad-exit \
torchrun --nnodes=1 --nproc_per_node=${numgpu} --master_port=25440 train.py \
    --stage 1 \
    --cfg ./config/LAMM/train_sft.yaml \
    --data_path  ../data/LAMM/2D_pretrain/meta_file/blip_laion_cc_sbu_558k.json \
    --vision_root_path ../data/LAMM/2D_pretrain/images \
    --conv_template default \
    --max_tgt_len 2048 \
    --vision_type image \
    --use_system \
    --model lamm_sft \
    --encoder_pretrain clip \
    --gradient_checkpointing \
    --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0/ \
    --vision_feature_type ${visfeat_type} \
    --num_vision_token 256 \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log

