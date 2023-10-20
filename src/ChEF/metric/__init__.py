from .classification import FG_Classification, CG_Classification
from .vqa import VQA, MMBenchVQA, MMEVQA
from .caption import Caption
from .desiderata import MMBench_Calibration, ScienceQA_Calibration, POPE_Metric, Instruct_Follow
from .detection import Detection, KOSMOS_Detection 
from .counting import Counting

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

}

def build_metric(metric_type, dataset_name, **kwargs):
    build_fuc = evaluation_protocol[metric_type][dataset_name]
    return build_fuc(dataset_name = dataset_name, **kwargs)