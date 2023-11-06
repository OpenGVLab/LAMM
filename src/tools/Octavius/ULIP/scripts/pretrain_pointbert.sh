current_time=`date "+%Y_%m_%d_%H_%M"`

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port 61234 \
main.py \
--model ULIP_PointBERT \
--npoints 2048 \
--lr 1e-4 \
--epochs 40 \
--batch_size 16 \
--lr_end 1e-5 \
--output_dir ./outputs/pointbert_2kpts_xyz_$current_time \
# --use_scanrefer \