# LAMM

Official Repository of [LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark]()

---
## News


## Demo



## Getting Started

### Installation


### Inference


### Training


## LAMM-Dataset 
![LAMM-Dataset](./images/LAMM-Dataset.png)
**LAMM-Dataset** includes an image instruction-tuning dataset containing **186,098** image-language instruction-response pairs and a point cloud instruction-tuning dataset with **10,262** point cloud-language instruction-response pairs. We collect images and point clouds from publicly available datasets and use the GPT API and self-instruction methods to generate instructions and responses based on the original labels from these datasets. The resulting LAMM-Dataset has three appealing properties: 
1) Existing multi-modal instruction tuning datasets mainly focus on holistic and rough information. To emphasize fine-grained and dense information, we add more visual information, such as visual relationships and fine-grained categories as input for the GPT API. 
2) We observe that existing MLLMs may struggle to understand vision task instructions. To address this, we designed a method to convert vision task annotations into instruction-response pairs, which enhances MLLMs' understanding and generalization of vision task instructions. 
3) LAMM-Dataset also includes data pairs for commonsense knowledge question answering by incorporating a hierarchical knowledge graph label system from the Bamboo dataset and the corresponding Wikipedia description.

## Data Download
<details><summary> LAMM-Dataset Directory Structure  </summary>
<p>

    ├── 2D_Instruct  
    │   ├── bamboo_images.zip  
    │   ├── coco_images.zip  
    │   ├── locount_images.zip  
    │   ├── textvqa_images.zip  
    │   ├── meta_file  
    │   │   ├── daily_dialogue_49k.json  
    │   │   ├── detailed_description_49k.json  
    │   │   ├── factual_knowledge_dialogue_42k.json  
    │   │   ├── LAMM_instruct_140k.json  
    │   │   ├── LAMM_instruct_186k.json  
    │   │   ├── LAMM_instruct_98k.json  
    │   │   └── vision_task_dialogue_46k.json  
        ├── 2D_Benchmark  
    │   ├── ai2d_images.zip  
    │   ├── celeba_images.zip  
    │   ├── cifar10_images.zip  
    │   ├── flickr30k_images.zip  
    │   ├── fsc147_images.zip  
    │   ├── lsp_images.zip  
    │   ├── sqaimage_images.zip  
    │   ├── svt_images.zip  
    │   ├── ucmerced_images.zip  
    │   ├── voc2012_images.zip  
    │   ├── meta_file  
    │   │   ├── Caption_flickr30k.json  
    │   │   ├── Classification_CIFAR10.json  
    │   │   ├── Counting_FSC147.json  
    │   │   ├── Detection_VOC2012.json  
    │   │   ├── Facial_Classification_CelebA(Hair).json  
    │   │   ├── Facial_Classification_CelebA(Smile).json  
    │   │   ├── Fine-grained_Classification_UCMerced.json  
    │   │   ├── Keypoints_Dectection_LSP.json  
    │   │   ├── Locating_FSC147.json  
    │   │   ├── Locating_LSP.json  
    │   │   ├── Locating_VOC2012.json  
    │   │   ├── OCR_SVT.json  
    │   │   ├── VQA_AI2D.json  
    │   │   └── VQA_SQAimage.json  
    ├── 3D_Instruct  
    │   ├── 3rscan_pcls.zip  
    │   ├── shapenet_pcls.zip  
    │   ├── meta_file  
    │   │   └── LAMM_3dinstruct_10k.json  
    └── 3D_Benchmark  
        ├── scannet_pcls.zip  
        ├── meta_file  
        │   ├── Detection_ScanNet.json  
        │   ├── VG_ScanRefer.json  
        │   └── VQA_ScanQA_multiplechoice.json   
</p>
</details>

<details><summary> LAMM-Dataset Files Download  </summary>
<p>

***
- ### 2D_Instruct data  

    |  Data file name  | size  |  
    |  ----  | ----  |  
    | [daily_dialogue_49k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/daily_dialogue_49k.json)  | 107M | 
    | [detailed_description_49k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/detailed_description_49k.json)  | 63M |
    | [factual_knowledge_dialogue_42k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/factual_knowledge_dialogue_42k.json) | 80M |
    | [vision_task_dialogue_46k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/vision_task_dialogue_46k.json) | 62M |
    | [LAMM_instruct_98k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/LAMM_2dinstruct_98k.json) | 170M |
    | [LAMM_instruct_140k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/LAMM_2dinstruct_140k.json) | 249M |
    | [LAMM_instruct_186k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/LAMM_2dinstruct_186k.json) | 311M |
    
    |  Image data  | size  |  
    |  ----  | ----  |  
    |  bamboo_images.zip  | 7.5G  |  
    |  coco_images.zip  | 8.5G  |  
    |  locount_images.zip  | 3.0G  |  
    |  textvqa_images.zip | 2.4G  |  
***
- ### 2D_Benchmark data  

    |  Data file name  | size  |  
    |  ----  | ----  |
    | [Caption_flickr30k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Caption_flickr30k.json)  | 598K |
    | [Classification_CIFAR10.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Classification_CIFAR10.json)  | 2.6M |
    | [Counting_FSC147.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Counting_FSC147.json) | 7.3M |
    | [Detection_VOC2012.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Detection_VOC2012.json) | 6.4M |
    | [Facial_Classification_CelebA(Hair).json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Facial_Classification_CelebA(Hair).json) | 2.4M |
    | [Facial_Classification_CelebA(Smile).json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Facial_Classification_CelebA(Smile).json) | 3.7M |
    | [Fine-grained_Classification_UCMerced.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Fine-grained_Classification_UCMerced.json) | 676K |
    | [Keypoints_Dectection_LSP.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Keypoints_Dectection_LSP.json) | 3.9M |
    | [Locating_FSC147.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Locating_FSC147.json) | 7.5M |
    | [Locating_LSP.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Locating_LSP.json) | 3.9M |
    | [Locating_VOC2012.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Locating_VOC2012.json) | 6.0M |
    | [OCR_SVT.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/OCR_SVT.json) | 68K |
    | [VQA_AI2D.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/VQA_AI2D.json) | 2.1M |
    | [VQA_SQAimage.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/VQA_SQAimage.json) | 3.6M |

    |  Image data  | size  |  
    |  ----  | ----  |  
    |  ai2d_images.zip  | 559M  |  
    |  celeba_images.zip  |  566M |  
    |  cifar10_images.zip  | 8.9M  |  
    |  flickr30k_images.zip  | 134M  |  
    | fsc147_images.zip   |  44M |  
    |  lsp_images.zip  |  9.9M |  
    |  sqaimage_images.zip  | 127M  |  
    |  svt_images.zip  | 82M  |  
    |  ucmerced_images.zip  | 317M  |  
    |  voc2012_images.zip  | 196M  |  


***
- ### 3D_Instruct data  
    |  Data file name  | size  |
    |  ----  | ----  |
    |  [LAMM_3dinstruct_10k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/3D_Instruct/meta_file/LAMM_3dinstruct_10k.json)  | 19M  |


    |  Image data  | size  |
    |  ----  | ----  |  
    |  3rscan_pcls.zip  | 720M  |   
    |  shapenet_pcls.zip  | 209M  | 
***
- ### 3D_Benchmark data  
    |  Data file name  | size  |
    |  ----  | ----  |   
    |  [Detection_ScanNet.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/3D_Benchmark/meta_file/Detection_ScanNet.json)  | 1.7M  |
    |  [VG_ScanRefer.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/3D_Benchmark/meta_file/VG_ScanRefer.json)  | 3.7M  |
    |  [VQA_ScanQA_multiplechoice.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/3D_Benchmark/meta_file/VQA_ScanQA_multiplechoice.json)  | 859K  |

    |  Image data  | size  |
    |  ----  | ----  |  
    |  scannet_pcls.zip  | 246M  |  
***
</p>
</details>

## Leaderboard


## Citation


## License & Acknowledgement
