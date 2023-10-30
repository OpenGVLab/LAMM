yaml_dict=(ScienceQA FSC147 VOC2012 SVT Flickr30k UCMerced  CelebA_hair CelebA_smile CIFAR10 AI2D locating_LSP locating_VOC2012)

for dataset in ${yaml_dict[*]}; do
    
    srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python eval.py --model_cfg config/ChEF/models/lamm.yaml --recipe_cfg config/ChEF/scenario_recipes/LAMM/${dataset}.yaml --debug

done