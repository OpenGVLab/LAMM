from .classification import FG_Classification, CG_Classification, LAMM_Classification, \
    LAMM_Facial_Hair_Classification, LAMM_Facial_Smile_Classification, LAMM_3D_Classification, \
        UCMerced_Classification
from .vqa import VQA, MMBenchVQA, MMEVQA, LAMM_VQA
from .caption import Caption, LAMM_Caption, LAMM_3D_Caption
from .desiderata import MMBench_Calibration, ScienceQA_Calibration, POPE_Metric, Instruct_Follow
from .detection import Detection, KOSMOS_Detection, LAMM_Detection , LAMM_3D_Detection, LAMM_3D_Grounding
from .counting import Counting
from .ocr import SVT_OCR
from .lamm_locating import InsideBbox, InsideHumanBbox
from .winoground import Winoground_Metric, Winoground_Cap_Metric

evaluation_protocol = {
    'basic':{
        'CIFAR10': CG_Classification,
        'Omnibenchmark': FG_Classification,
        'Flickr30k' : Caption,
        'ScienceQA': VQA,
        'VOC2012': Detection,
        'FSC147': Counting,
        'MMBench': MMBenchVQA,
        'MME': MMEVQA,
        'SEEDBench': VQA
    },
    'Calibration':
    {
        'ScienceQA': ScienceQA_Calibration,
        'MMBench': MMBench_Calibration
    },
    'Hallucination':
    {
        'POPE_COCO_random': POPE_Metric,
        'POPE_COCO_popular': POPE_Metric,
        'POPE_COCO_adversarial': POPE_Metric,
    },
    'Instruct_Follow':
    {
      'ScienceQA': Instruct_Follow,
      'MMBench': Instruct_Follow,
    },
    'KOSMOS':{ # kosmos outputs special tokens for bbox
        'VOC2012': KOSMOS_Detection,
    },
    'LAMM': {
        'VOC2012': LAMM_Detection,
        'Flickr30k': LAMM_Caption,
        'ScienceQA': LAMM_VQA,
        'CIFAR10': LAMM_Classification,
        'CelebA(Hair)': LAMM_Facial_Hair_Classification,
        'CelebA(Smile)': LAMM_Facial_Smile_Classification,
        'SVT': SVT_OCR,
        'UCMerced': UCMerced_Classification,
        'AI2D': LAMM_VQA,
        'ScanQA_LAMM': LAMM_VQA,
        'ScanNet_LAMM': LAMM_3D_Detection, 
        'ScanRefer_LAMM': LAMM_3D_Grounding,
        'Locating_VOC2012': InsideBbox,
        'Locating_LSP': InsideHumanBbox
    },
    'Octavius3D': {
        'scannet_Classification': LAMM_3D_Classification,
        'scannet_Caption': LAMM_3D_Caption,
        'scannet_VQA': LAMM_3D_Caption,
        'nr3d_Caption': LAMM_3D_Caption,
        'shapenet_Classification': LAMM_3D_Classification,
    },
    'Winoground': {
        'Winoground': Winoground_Metric,
        'Winoground_Cap': Winoground_Cap_Metric,
    }
}

def build_metric(metric_type, dataset_name, **kwargs):
    build_func = evaluation_protocol[metric_type][dataset_name]
    return build_func(dataset_name = dataset_name, **kwargs)