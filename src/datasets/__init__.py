from header import *
from .samplers import DistributedBatchSampler
from .dataset import *


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
