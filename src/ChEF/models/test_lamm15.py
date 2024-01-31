import torch
from .test_lamm import TestLAMM
from model.LAMM import LAMMSFTModel

class TestLAMM15(TestLAMM):
    def __init__(self, model_path, task_type='normal', **kwargs):
        self.conv_mode = 'simple'
        self.model = LAMMSFTModel(**kwargs)
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt, strict=False)             # TODO: load delta_ckpt from model_path in lamm_3d.yaml
        self.model = self.model.eval().half()
        self.task_type = task_type
        self.move_to_device()