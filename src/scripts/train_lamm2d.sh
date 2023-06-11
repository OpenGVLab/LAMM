#!/bin/bash
numgpu=4

exp=$1
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p ../ckpt/${exp}/log_rest/
deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 28457 train.py \
    --stage 1 \
    --cfg ./config/train.yaml \
    --data_path  ../data/2D_Instruct/meta_file/LAMM_instruct_186k.json \
    --vision_root_path ../data/2D_Instruct/ \
    --max_tgt_len 400 \
    --vision_type image \
    --use_system \
    --model lamm_peft \
    --encoder_pretrain clip \
    --vicuna_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0/ \
    --vision_feature_type ${visfeat_type} \
    --num_vision_token 256 \
    --save_path  ../ckpt/${exp} \
    --log_path ../ckpt/${exp}/log_rest/ \
    2>&1 | tee ../ckpt/${exp}/log_rest/train_${now}.log
