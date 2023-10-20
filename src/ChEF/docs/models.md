# MLLMs


Supported MLLMs in ChEF:

- [x] [InstructBLIP](https://github.com/salesforce/LAVIS)
- [x] [Kosmos2](https://github.com/microsoft/unilm/tree/master/kosmos-2)
- [x] [LAMM](https://github.com/OpenLAMM/LAMM)
- [x] [LLaMA-Adapter-v2](https://github.com/ml-lab/LLaMA-Adapter-2)
- [x] [LLaVA](https://github.com/haotian-liu/LLaVA)
- [x] [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [x] [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl)
- [x] [Otter](https://github.com/Luodian/Otter)
- [x] [Shikra](https://github.com/shikras/shikra)

## Checkpoints
The checkpoints evaluated in ChEF are listed as follows: 

| LLM              | Vision Encoder | Language Model | Link                                                              |
| :--------:       | :--------:     | :------------: | :---------------------------------------------------------------: |
| InstructBLIP     | EVA-G          |    Vicuna 7B   | [instruct_blip_vicuna7b_trimmed](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth) |
| Kosmos2          | CLIP ViT-L/14  |  Decoder 1.3B  | [kosmos-2.pt](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/kosmos-2.pt?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) |
| LAMM             | CLIP ViT-L/14  |  Vicuna 13B    | [lamm_13b_lora32_186k](https://huggingface.co/openlamm/lamm_13b_lora32_186k) |
| LLaMA-Adapter-v2 | CLIP ViT-L/14  |    LLaMA 7B    | [LORA-BIAS-7B](https://github.com/ml-lab/LLaMA-Adapter-2) |
| LLaVA            | CLIP ViT-L/14  |    MPT 7B      | [LLaVA-Lightning-MPT-7B](https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview) |
| MiniGPT-4        | EVA-G          |   Vicuna 7B    | [MiniGPT-4](https://huggingface.co/Vision-CAIR/MiniGPT-4) |
| mPLUG-Owl        | CLIP ViT-L/14  |    LLaMA 7B    | [mplug-owl-llama-7b](https://huggingface.co/MAGAer13/mplug-owl-llama-7b) |
| Otter            | CLIP ViT-L/14  |    LLaMA 7B    | [OTTER-9B-LA-InContext](https://huggingface.co/luodian/OTTER-Image-LLaMA7B-LA-InContext) |
| Shikra           | CLIP ViT-L/14  |    LLaMA 7B    | [shikra-7b](https://huggingface.co/shikras/shikra-7b-delta-v1) |

## Requirements
Please refer to the respective repositories of each model for their requirements.

## Usage
Define the model configs in [models](../configs/models/), including `model_name`, `model_path` and other neccessary configs. For certain models, different configurations are required when testing on different recipes. For example, the default config for [KOSMOS-2](../configs/models/kosmos2.yaml):
```yaml
model_name: Kosmos2
model_path: data/checkpoints/kosmos/kosmos-2.pt
if_grounding: False
``` 
The config for KOSMOS-2 on detection tasks evaluation:

```yaml
model_name: Kosmos2
model_path: data/checkpoints/kosmos/kosmos-2.pt
if_grounding: True
``` 