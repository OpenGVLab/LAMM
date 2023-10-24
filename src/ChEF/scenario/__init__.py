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


dataset_dict = {
    # Caption 
    'Flickr30k': FlickrDataset,
    'Flickr30k_LAMM': FlickrLAMMDataset,
    # classification
    'CIFAR10': CIFAR10Dataset,
    'Omnibenchmark': OmnibenchmarkDataset,
    'CIFAR10_LAMM' : CIFAR10LAMMDataset,
    # VQA 
    'ScienceQA': ScienceQADataset,
    'ScienceQA_LAMM' : ScienceQALAMMDataset,
    # Detection
    'VOC2012': VOC2012Dataset,
    'VOC2012_LAMM': VOC2012LAMMDataset,
    # Counting
    'FSC147': FSC147Dataset,
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
    # Facial Classification
    'CelebA(Hair)' : CelebAHairDataset,
    'CelebA(Smile)' : CelebASmileDataset,
    # 3D
    'OctaviusPCLDataset': OctaviusPCLDataset,
}
