#!/bin/bash
numgpu=4

exp=$1
visfeat_type=local
now=$(date +"%Y%m%d_%H%M%S")

ckpt_dir=../ckpt
mkdir -p ${ckpt_dir}/${exp}/log_rest/
deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 28457 train.py \
    --stage 1 \
    --cfg ./config/train.yaml \
    --data_path  ../data/LAMM/3D_Instruct/meta_file/LAMM_3dinstruct_10k.json \
    --vision_root_path ../data/LAMM/3D_Instruct/ \
    --max_tgt_len 400 \
    --vision_type pcl \
    --use_system \
    --model lamm_peft \
    --encoder_pretrain epcl \
    --encoder_ckpt_path ../model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth \
    --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0/ \
    --vision_feature_type ${visfeat_type} \
    --num_vision_token 256 \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log
