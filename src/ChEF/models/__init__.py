import torch

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

def get_model(cfg, device):
    model_name = cfg['model_name']
    if model_name == 'InstructBLIP':
        from .test_instructblip import TestInstructBLIP
        return TestInstructBLIP(device=device)
    elif model_name == 'LLaMA-Adapter-v2':
        from .test_llamaadapterv2 import TestLLamaAdapterV2
        return TestLLamaAdapterV2(device=device, **cfg)
    elif model_name == 'MiniGPT-4':
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(device=device, **cfg)
    elif model_name == 'mPLUG-Owl':
        from .test_mplugowl import TestMplugOwl
        return TestMplugOwl(device=device,**cfg)
    elif model_name == 'Otter':
        from .test_otter import TestOtter
        return TestOtter(device=device, **cfg)
    elif model_name == 'Kosmos2':
        from .test_kosmos import TestKOSMOS2
        return TestKOSMOS2(device=device,**cfg)
    elif model_name == 'LAMM':
        from .test_lamm import TestLAMM
        return TestLAMM(device=device, **cfg)
    elif model_name == 'LAMM_SFT' :
        from .test_lamm15 import TestLAMM15
        return TestLAMM15(device=device, **cfg)
    elif model_name == 'Octavius' or model_name == 'Octavius_3d' or model_name == 'Octavius_2d':
        from .test_octavius import TestOctavius
        return TestOctavius(**cfg)
    elif model_name == 'Shikra':
        from .test_shikra import TestShikra
        return TestShikra(device=device,**cfg)
    elif model_name == 'LLaVA1.5':
        from .test_llava15 import TestLLaVA15
        return TestLLaVA15(device=device, **cfg)
    elif model_name == 'LLaVARLHF':
        from .test_llavarlhf import TestLLaVARLHF
        return TestLLaVARLHF(device=device, **cfg)
    elif model_name == 'InternLMXComposer':
        from .test_internlmxcomposer import TestInternlmXcomposer
        return TestInternlmXcomposer(device=device, **cfg)
    elif model_name == 'QwenVL':
        from .test_qwenvl import TestQwenVL
        return TestQwenVL(device=device, **cfg)
    elif model_name == 'Test':
        from .test_base import TestBase
        return TestBase(**cfg)
    elif model_name == 'RLHFV':
        from .test_rlhfv import TestRLHFV
        return TestRLHFV(device=device, **cfg)
    elif model_name == 'GPT':
        from .test_gpt import TestGPT
        return TestGPT(**cfg)
    elif model_name == 'Gemini':
        from .test_gemini import TestGemini
        return TestGemini(**cfg)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
