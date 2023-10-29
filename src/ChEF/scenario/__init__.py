from .caption_dataset import *
from .classification import *
from .vqa_dataset import *
from .det_dataset import *
from .counting_dataset import *

from .MMBench_dataset import *
from .SEED_Bench_dataset import *
from .MME_dataset import *

from .POPE_dataset import *

from .octavius_pcl_dataset import OctaviusPCLDataset

from .LAMM_dataset import *


dataset_dict = {
    # Caption 
    'Flickr30k': FlickrDataset,
    # classification
    'CIFAR10': CIFAR10Dataset,
    'Omnibenchmark': OmnibenchmarkDataset,
    # VQA 
    'ScienceQA': ScienceQADataset,
    # Detection
    'VOC2012': VOC2012Dataset,
    # Counting
    'FSC147': FSC147Dataset,
    
    # LAMM
    'CIFAR10_LAMM' : CIFAR10LAMMDataset,
    'VOC2012_LAMM': VOC2012LAMMDataset,
    'ScienceQA_LAMM' : ScienceQALAMMDataset,
    'Flickr30k_LAMM': FlickrLAMMDataset,
    'SVT': SVTDataset,
    'FSC147_LAMM': FSC147LAMMDataset,
    'UCMerced': UCMercedDataset,
    'CelebA(Hair)' : CelebAHairDataset,
    'CelebA(Smile)' : CelebASmileDataset,
    'AI2D': AI2DDataset,
    'ScanQA_LAMM': ScanQALAMMDataset,

    # MMBench
    'MMBench': MMBenchDataset,
    # Hallucination
    'POPE_COCO_random':POPE_COCO_Random_Dataset,
    'POPE_COCO_popular':POPE_COCO_Popular_Dataset,
    'POPE_COCO_adversarial':POPE_COCO_Adversarial_Dataset,
    # SEEDBench
    'SEEDBench': SEEDBenchDataset,
    # MME
    'MME': MMEDataset,
    
    # 3D
    'OctaviusPCLDataset': OctaviusPCLDataset,
}
