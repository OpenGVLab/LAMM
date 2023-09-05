common_dataset=(LSP SQAimage FSC147 VOC2012 SVT flickr30k UCMerced  CelebA\(Hair\) CelebA\(Smile\) CIFAR10 AI2D)
locating_dataset=(VOC2012 LSP FSC147)
exp=lamm_13b_lora_186k
base_data_path=../data/2D_Benchmark
token_num=256
layer=-2
answerdir=../answers
mkdir -p ${answerdir}/${exp}
results_path=../results
mkdir -p ${results_path}/${exp}


for dataset in ${common_dataset[*]}; do

    python inference_2d.py \
        --model lamm_peft \
        --encoder_pretrain clip \
        --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0 \
        --delta_ckpt_path ../ckpt/${exp}/pytorch_model.pt \
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
    
    python common_eval_2d.py \
    --dataset-name ${dataset} \
    --answer-file ${answerdir}/${exp} \
    --base-data-path ${base_data_path} \
    2>&1 | tee ${results_path}/${exp}/eval_${dataset}.log
done

for dataset in ${locating_dataset[*]}; do
    python inference_2d.py \
        --model lamm_peft \
        --encoder_pretrain clip \
        --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0 \
        --delta_ckpt_path ../ckpt/${exp}/pytorch_model.pt \
        --max_tgt_len 400 \
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --num_vision_token ${token_num} \
        --vision_output_layer ${layer} \
        --conv_mode simple \
        --dataset-name ${dataset} \
        --base-data-path ${base_data_path} \
        --inference-mode locating \
        --bs 32 \
        --answers-dir ${answerdir}/${exp} \

done

python locating_eval.py \
    --answer-dir ${answerdir}/${exp} \
    --base-data-path ${base_data_path} \
    2>&1 | tee ${results_path}/${exp}/eval_locating.log