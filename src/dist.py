import os
import socket
import time
import torch

def init_distributed_mode():
    if 'SLURM_PROCID' in os.environ:
        global_rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        local_rank = global_rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return dict(
            local_rank=0, 
            global_rank=0, 
            world_size=1)
    
    print(f"Start inference, world_size: {world_size}, global_rank: {global_rank}, local_rank:{local_rank}")
    os.environ['LOCAL_RANK'] = str(local_rank)

    return dict(
        local_rank=local_rank, 
        global_rank=global_rank, 
        world_size=world_size)