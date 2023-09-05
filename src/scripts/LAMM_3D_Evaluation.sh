exp=$1
common_dataset=(ScanNet ScanRefer ScanQA_multiplechoice)
base_data_path=../data/3D_Benchmark
visfeat_type=local
token_num=256
layer=-2

answerdir=../answers
mkdir -p ${answerdir}/${exp}
results_path=../results
mkdir -p ${results_path}/${exp}


for dataset in ${common_dataset[*]}; do

    python inference_3d.py \
        --model lamm_peft \
        --encoder_pretrain epcl \
        --encoder_ckpt_path ../model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth \
        --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0 \
        --delta_ckpt_path ../ckpt/${exp}/pytorch_model.pt \
        --max_tgt_len 800 \
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --vision_feature_type ${visfeat_type} \
        --num_vision_token ${token_num} \
        --vision_output_layer ${layer} \
        --conv_mode simple \
        --dataset-name ${dataset} \
        --base-data-path ${base_data_path} \
        --inference-mode common \
        --bs 1 \
        --answers-dir ${answerdir}/${exp} \
    
    python common_eval_3d.py \
        --dataset-name ${dataset} \
        --answer-file ${answerdir}/${exp} \
        --base-data-path ${base_data_path} \
        2>&1 | tee ${results_path}/${exp}/eval_${dataset}.log
done
