from header import *
from .samplers import DistributedBatchSampler
from .dataset import *
from .LAMM_banchmark2D_dataset import LAMM_EVAL_2D

def load_lamm_dataset(args):
    """load LAMM datasets

    :param dict args: input arguments
    :return tupe: dataset, dataloader, sampler
    """
    dataset = LAMMDataset(
        args["data_path"], args["vision_root_path"], args["vision_type"]
    )

    sampler = torch.utils.data.RandomSampler(dataset)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = (
        args["world_size"] * args["dschf"].config["train_micro_batch_size_per_gpu"]
    )
    batch_sampler = DistributedBatchSampler(sampler, batch_size, True, rank, world_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=1,
        collate_fn=dataset.collate,
        pin_memory=True,
    )
    return dataset, dataloader, sampler



def load_2Deval_dataset(base_data_path,
                        dataset_name,
                        mode = 'common',
                        load_img = True,
                        batch_size = 1):
    dataset = LAMM_EVAL_2D(base_data_path,
                           dataset_name,
                           mode,
                           load_img)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=False,num_workers=2, drop_last=False,collate_fn=collate_fn)
    return dataloader