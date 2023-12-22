import torch

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

def get_model(cfg):
    model_name = cfg['model_name']
    if model_name == 'InstructBLIP':
        from .test_instructblip import TestInstructBLIP
        return TestInstructBLIP()
    elif model_name == 'InstructBLIPT5':
        from .test_instructblip_t5 import TestInstructBLIPT5
        return TestInstructBLIPT5()
    elif model_name == 'LLaMA-Adapter-v2':
        from .test_llamaadapterv2 import TestLLamaAdapterV2
        return TestLLamaAdapterV2(**cfg)
    elif model_name == 'LLaVA':
        from .test_llava import TestLLaVA
        return TestLLaVA(**cfg)
    elif model_name == 'MiniGPT-4':
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(**cfg)
    elif model_name == 'mPLUG-Owl':
        from .test_mplugowl import TestMplugOwl
        return TestMplugOwl(**cfg)
    elif model_name == 'Otter':
        from .test_otter import TestOtter
        return TestOtter(**cfg)
    elif model_name == 'Kosmos2':
        from .test_kosmos import TestKOSMOS2
        return TestKOSMOS2(**cfg)
    elif model_name == 'LAMM':
        from .test_lamm import TestLAMM
        return TestLAMM(**cfg)
    elif model_name == 'Octavius' or model_name == 'Octavius_3d' or model_name == 'Octavius_2d':
        from .test_octavius import TestOctavius
        return TestOctavius(**cfg)
    elif model_name == 'Shikra':
        from .test_shikra import TestShikra
        return TestShikra(**cfg)
    elif model_name == 'Test':
        from .test_base import TestBase
        return TestBase(**cfg)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
