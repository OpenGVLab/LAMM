model_cfg=config/ChEF/models/octavius_2d+3d.yaml
recipe_cfg_list=(CIFAR10 Flickr30k CelebA_hair CelebA_smile VOC2012 ScienceQA)


for dataset in ${recipe_cfg_list[*]}; do
    srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
        python eval.py \
            --model_cfg ${model_cfg} \
            --recipe_cfg config/ChEF/scenario_recipes/LAMM/${dataset}.yaml --debug

done


recipe_cfg_list=(nr3d_caption_direct3d scan_caption_direct3d scan_cls_direct3d scan_vqa_direct3d shapenet_cls_direct3d)


for dataset in ${recipe_cfg_list[*]}; do
    srun -p AI4Good_X --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
        python eval.py \
            --model_cfg ${model_cfg} \
            --recipe_cfg config/ChEF/scenario_recipes/Octavius3D/${dataset}.yaml --debug

done