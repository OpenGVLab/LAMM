from .Direct import Direct_inferencer, Det_Direct_inferencer, Icl_Direct_inferencer
from .PPL import PPL_inferencer, ICL_PPL_inferencer, Det_PPL_inferencer, Cali_inferencer
from .Multiturn import Multi_Turn_PPL_inferencer

inferencer_dict = {
    'Direct': Direct_inferencer,
    'Det': Det_Direct_inferencer,
    'Det_PPL': Det_PPL_inferencer,
    'PPL': PPL_inferencer,
    'Multi_PPL': Multi_Turn_PPL_inferencer,
    'ICL_Direct': Icl_Direct_inferencer,
    'Calibration':Cali_inferencer,
    'ICL_PPL': ICL_PPL_inferencer
}

def build_inferencer(inferencer_type, **kwargs):
    return inferencer_dict[inferencer_type](**kwargs)