# LAMM Dataset Overview
LAMM-Dataset is a comprehensive multi-modal instruction tuning dataset, which contains 186K language-image instruction-response pairs, and 10K lanuage-3D instruction-response pairs.In LAMM-Dataset, the instruction-response pairs are gathered from 8 image datasets and 4 point cloud datasets. Here we design four kinds of multi-modal instruction-response pairs, 
- C1: n-round daily dialogue focuses on multi-modal daily conversations. 
- C2: n-round factual knowledge dialogue aims at factual knowledge reasoning. 
- C3: 1-round detailed description aims to elaborate images and 3D scenes in texts. 
- C4: 1-round visual task dialogue transfers various vision tasks into instruction-response pairs, aiming at enhancing generalizability towards domain tasks in other modalities.


#  Dataset Statistics
Download LAMM-Dataset from [here](https://opendatalab.com/LAMM/download).
- ## 2D_Instruct data  

    <!-- |  Data file name  | size  |  Image file name | size |
    |  ----  | ----  |  ----  | ----  |  
    | [daily_dialogue_49k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/daily_dialogue_49k.json)  | 107M | daily_dialogue_description_images.zip | 7.8G |  
    | [detailed_description_49k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/detailed_description_49k.json)  | 63M |  daily_dialogue_description_images.zip | 7.8G |   
    | [factual_knowledge_dialogue_42k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/factual_knowledge_dialogue_42k.json) | 80M | factual_knowledge_dialogue_images.zip | 5.4G |
    | [vision_task_dialogue_46k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/vision_task_dialogue_46k.json) | 62M | vision_task_dialogue_images.zip |9.2G |   -->
    <!-- | [LAMM_instruct_98k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/LAMM_2dinstruct_98k.json) | 170M |
    | [LAMM_instruct_140k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/LAMM_2dinstruct_140k.json) | 249M |
    | [LAMM_instruct_186k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/LAMM_2dinstruct_186k.json) | 311M | -->

    |  Data file name  | size  |  Image file name |  size |  
    |  ----  | ----  |  ----  | ---- |   
    | [daily_dialogue_49k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/daily_dialogue_49k.json)  | 107M | coco_images.zip | 7.8G |   
    | [detailed_description_49k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/detailed_description_49k.json)  | 63M |  coco_images.zip | 7.8G |    
    | [factual_knowledge_dialogue_42k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/factual_knowledge_dialogue_42k.json) | 80M | bamboo_images.zip | 5.4G |  
    | [vision_task_dialogue_46k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Instruct/meta_file/vision_task_dialogue_46k.json) | 62M | coco_images.zip, bamboo_images.zip, locount_images.zip, textvqa_images.zip | 9.2G |  

- ## 2D_Benchmark data  

    |  Data file name  | size  |  Image file name | size |  
    |  ----  | ----  |  ----  | ----  |  
    | [Caption_flickr30k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Caption_flickr30k.json)  | 598K | flickr30k_images.zip | 559M |     
    | [Classification_CIFAR10.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Classification_CIFAR10.json)  | 2.6M | cifar10_images.zip  | 8.9M  |  
    | [Counting_FSC147.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Counting_FSC147.json) | 7.3M | fsc147_images.zip   |  44M |  
    | [Detection_VOC2012.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Detection_VOC2012.json) | 6.4M | voc2012_images.zip  | 196M  |  
    | [Facial_Classification_CelebA(Hair).json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Facial_Classification_CelebA(Hair).json) | 2.4M | celeba_images.zip  |  566M |  
    | [Facial_Classification_CelebA(Smile).json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Facial_Classification_CelebA(Smile).json) | 3.7M |  celeba_images.zip  |  566M |  
    | [Fine-grained_Classification_UCMerced.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Fine-grained_Classification_UCMerced.json) | 676K | ucmerced_images.zip  | 317M  |  
    | [Keypoints_Dectection_LSP.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Keypoints_Dectection_LSP.json) | 3.9M |  fsc147_images.zip   |  44M |   
    | [Locating_FSC147.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Locating_FSC147.json) | 7.5M | fsc147_images.zip   |  44M |  
    | [Locating_LSP.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Locating_LSP.json) | 3.9M | lsp_images.zip  |  9.9M |  
    | [Locating_VOC2012.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/Locating_VOC2012.json) | 6.0M | voc2012_images.zip  | 196M  |  
    | [OCR_SVT.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/OCR_SVT.json) | 68K |  svt_images.zip  | 82M  |  
    | [VQA_AI2D.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/VQA_AI2D.json) | 2.1M | ai2d_images.zip  | 559M  |  
    | [VQA_SQAimage.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/2D_Benchmark/meta_file/VQA_SQAimage.json) | 3.6M |  sqaimage_images.zip  | 127M  |  

- ## 3D_Instruct data  
    |  Data file name  | size  |  Image file name  | size  |  
    |  ----  | ----  | ----  | ----  | 
    |  [LAMM_3dinstruct_10k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/3D_Instruct/meta_file/LAMM_3dinstruct_10k.json)  | 19M  | 3rscan_pcls.zip  | 720M  |  
    |  [LAMM_3dinstruct_10k.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/3D_Instruct/meta_file/LAMM_3dinstruct_10k.json)  | 19M  | shapenet_pcls.zip  | 209M  |  

- ## 3D_Benchmark data  
    |  Data file name  | size  |  Image file name  | size  |  
    |  ----  | ----  |  ----  | ----  |   
    |  [Detection_ScanNet.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/3D_Benchmark/meta_file/Detection_ScanNet.json)  | 1.7M  | scannet_pcls.zip  | 246M  | 
    |  [VG_ScanRefer.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/3D_Benchmark/meta_file/VG_ScanRefer.json)  | 3.7M  | scannet_pcls.zip  | 246M  | 
    |  [VQA_ScanQA_multiplechoice.json](https://huggingface.co/datasets/caojianjian/LAMM/blob/main/3D_Benchmark/meta_file/VQA_ScanQA_multiplechoice.json)  | 859K  | scannet_pcls.zip  | 246M  | 

### Note: The LAMM dataset is stored in the following format:
```json
{
    "id": "000000019028",  # image id
    "image": "coco_images/000000019028.jpg", # image path
    "conversations": [
        {
            "from": "human",  # instruction
            "value": "How is the kitchen in the image furnished?"
        },
        {
            "from": "gpt",  # response
            "value": "The kitchen in the image is furnished with white cabinets and white appliances. There is a dishwasher, a stove, and a sink. On the stove, a blue towel hangs on the handle. A cutting board is placed on the dishwasher. There are also additional elements like a bowl of apples on the counter and a beige rug on the floor."
        }
    ],
    "task_type": "conversation",  # task type
    "src_image": "coco2017" # original dataset
}
```

# LAMM-Dataset Directory Structure 
    ├── 2D_Instruct  
    │   ├── coco_images.zip  
    │   ├── bamboo_images.zip  
    │   ├── textvqa_images.zip  
    │   ├── locount_images.zip  
    │   ├── meta_file  
    │   │   ├── daily_dialogue_49k.json  
    │   │   ├── detailed_description_49k.json  
    │   │   ├── factual_knowledge_dialogue_42k.json  
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
