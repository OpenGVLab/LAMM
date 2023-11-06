<!-- # 🐏LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark -->

![LAMM](./images/lamm-title.png)
<p align="center">
    <font size='4'>
    <a href="https://openlamm.github.io/" target="_blank">🌏 Project Page</a> • <a href="https://openxlab.org.cn/apps/detail/LAMM/LAMM" target="_blank">𝕏 Demo</a> • <a href="https://www.youtube.com/watch?v=M7XlIe8hhPk" target="_blank">▶️ YouTube </a> • <a href="https://www.bilibili.com/video/BV1kN411D7kt/?share_source=copy_web&vd_source=ab4c734425ed0114898300f2c037ac0b" target="_blank"> 📺 Bilibili <a href="https://opendatalab.com/LAMM" target="_blank">📀 Data</a> • <a href="https://github.com/OpenLAMM/LAMM#lamm-benchmark" target="_blank">📊 Benchmark</a> • <a href="https://huggingface.co/openlamm" target="_blank">📦 LAMM Models</a>
    </font>
</p>


### Official Repository of [LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark](https://arxiv.org/abs/2306.06687)

[![](./images/lamm-video.png)](https://www.youtube.com/watch?v=M7XlIe8hhPk)

# Updates
📆 [**2023-09-02**]

1. Deepspeed ZeRO stage3, [xformers](https://github.com/facebookresearch/xformers) & [flashattention v2 (V100 not available)](https://github.com/Dao-AILab/flash-attention) are available for efficient training. LightLLM enabled for inference. Training your models on **V100** or **RTX3090**!
2. Checkpoints finetuning LLaMA2 online, please check it out.
3. Our demo is moved to [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/LAMM/LAMM), and previous session on Huggingface will be discarded.

📆 [**2023-07-28**]

1. Checkpoints of LAMM on [huggingface](https://huggingface.co/openlamm) updated on new code base. [LAMM Performance](#leaderboard) updated, Please check it out.

📆 [**2023-07-06**]

1. Evaluation code for both 2D and 3D tasks are ready.
2. [3D Benchmark meta files](https://huggingface.co/datasets/openlamm/LAMM_Dataset/tree/main/3D_Benchmark/meta_file) & [2D Instruction images files](https://opendatalab.com/LAMM/download) updated! Missing files and missing keys fixed. Please update accordingly.
3. Update scripts for [demo in command line](./src/cli_demo.py).

📆 [**2023-06-30**]

1. Watch demo video for LAMM at [Youtube](https://www.youtube.com/watch?v=M7XlIe8hhPk) and [Bilibili](https://www.bilibili.com/video/BV1kN411D7kt/?share_source=copy_web&vd_source=ab4c734425ed0114898300f2c037ac0b)!

📆 [**2023-06-20**]

1. [Full paper with Appendix](https://arxiv.org/abs/2306.06687) is online.

📆 [**2023-06-16**]

1. [LAMM dataset](#lamm-dataset) is available for Research community!

📆 [**2023-06-12**]

1. GPT Evaluation part available.

2. Our Paper will release tomorrow. Please stay tuned!

📆 [**2023-06-11**]

1. LAMM code is available for Research community!

2. Try out the [Interactive Demo](https://huggingface.co/spaces/openlamm/LAMM) on Huggingface! (Time to build app depends on the server load)


# Demos

## Online Demo

For cases of 2D images, we provide an [online demo](https://huggingface.co/spaces/openlamm/LAMM) deployed on huggingface spaces.

```
Due to limitation of hardware capacity, online version only supports LLM of 7B parameters and load pretrained model takes few minutes.
```

<!-- <a href="https://huggingface.co/spaces/openlamm/LAMM"><p align="center"><img src="./images/LAMM_2d_demo.png" height=600px/></p></a>  -->
[![](./images/LAMM_2d_demo.png)](https://huggingface.co/spaces/openlamm/LAMM)

## CLI Demo

We also provide a CLI demo for local test. 
Point cloud data are required to be in format of `npy`, we suggest to use data from LAMM-Benchmark-3D.


```bash
    cd ./src
    python cli_demo.py \
        --model lamm_peft \
        --vision_type pcl or image \
        --encoder_pretrain epcl or clip \
        --encoder_ckpt_path $EPCL_CKPT_PATH or '' \
        --llm_ckpt_path $LLM_CKPT_PATH \
        --delta_ckpt_path $LAMM_CKPT_PATH
```



# Overview
Large language models have become a potential pathway toward achieving artificial general intelligence. Recent works on multi-modal large language models have demonstrated their effectiveness in handling visual modalities. In this work, we extend the research of MLLMs to point clouds and present the LAMM-Dataset and LAMM-Benchmark for 2D image and 3D point cloud understanding. We also establish an extensible framework to facilitate the extension of MLLMs to additional modalities.
Our main contribution is three-fold: 1) We present the LAMM-Dataset and LAMM-Benchmark, which cover almost all high-level vision tasks for 2D and 3D vision. Extensive experiments validate the effectiveness of our dataset and benchmark. 2) We demonstrate the detailed methods of constructing instruction-tuning datasets and benchmarks for MLLMs, which will enable future research on MLLMs to scale up and extend to other domains, tasks, and modalities faster. 3) We provide a primary but potential MLLM training framework optimized for modalities' extension. We also provide baseline models, comprehensive experimental observations, and analysis to accelerate future research. 

# LAMM-Dataset

LAMM-Dataset is a comprehensive multi-modal instruction tuning dataset, which contains 186K language-image instruction-response pairs, and 10K lanuage-3D instruction-response pairs.In LAMM-Dataset, the instruction-response pairs are gathered from 8 image datasets and 4 point cloud datasets. Here we design four type of multi-modal instruction-response pairs, 
- C1: n-round daily dialogue focuses on multi-modal daily conversations. 
- C2: n-round factual knowledge dialogue aims at factual knowledge reasoning. 
- C3: 1-round detailed description aims to elaborate images and 3D scenes in texts. 
- C4: 1-round visual task dialogue transfers various vision tasks into instruction-response pairs, aiming at enhancing generalizability towards domain tasks in other modalities.


# Download 

Download LAMM-Dataset from [here](https://opendatalab.com/LAMM/download).

 If you would like to download the entire LAMM Dataset and LAMM Benchmark, you can do so from the opendatalab website using the provided [LAMM link](https://opendatalab.com/LAMM/download). Here is the table illustrating the correspondence between each Meta file and image collection in the LAMM dataset:
<details><summary> Instruction Data For Training</summary>

- 2D_Instruct data 

    |  Meta file name  | size  |  Image file name |  size |  
    |  ----  | ----  |  ----  | ---- |   
    | [daily_dialogue_49k.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Instruct/meta_file/daily_dialogue_49k.json)  | 112M | coco_images.zip | 7.8G |   
    | [detailed_description_49k.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Instruct/meta_file/detailed_description_49k.json)  | 65.5M |  coco_images.zip | 7.8G |    
    | [factual_knowledge_dialogue_42k.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Instruct/meta_file/factual_knowledge_dialogue_42k.json) | 83.2M | bamboo_images.zip | 5.4G |  
    | [vision_task_dialogue_46k.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Instruct/meta_file/vision_task_dialogue_46k.json) | 64.8M | coco_images.zip, bamboo_images.zip, locount_images.zip, textvqa_images.zip | 9.2G |  

- 3D_Instruct data

    |  Meta file name  | size  |  Image file name  | size  |  
    |  ----  | ----  | ----  | ----  | 
    |  [LAMM_3dinstruct_10k.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/3D_Instruct/meta_file/LAMM_3dinstruct_10k.json)  | 19.6M  | 3rscan_pcls.zip  | 720M  |  
    |  |  | shapenet_pcls.zip  | 209M  |  

</p>
</details> 


<details><summary> Dataset Structure </summary>

    └── 2D_Instruct  
    │   ├── coco_images.zip  
    │   ├── bamboo_images.zip  
    │   ├── textvqa_images.zip  
    │   ├── locount_images.zip  
    │   └── meta_file  
    │       ├── daily_dialogue_49k.json  
    │       ├── detailed_description_49k.json  
    │       ├── factual_knowledge_dialogue_42k.json  
    │       └── vision_task_dialogue_46k.json  
    └── 3D_Instruct  
        ├── 3rscan_pcls.zip  
        ├── shapenet_pcls.zip  
        └── meta_file  
            └── LAMM_3dinstruct_10k.json  

</p>
</details> 

<details><summary> Meta file format </summary>

- For images
```json
[
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
    },
    {
        ...
    }
]
```
- For point cloud
```json
[
    {
        "pcl": "shapenet_pcls/04256520_cb71cb7b36dbcb6f826fc8d57346a2e4_4096.npy",
        "conversations": [
                {
                    "from": "human",
                    "value": "What scenario does this point cloud belong to according to the model\u2019s prediction?"
                },
                {
                    "from": "gpt",
                    "value": "Through meticulous analysis, it becomes evident that the point cloud aligns with the characteristics of sofa,couch,lounge s       cenario."
                }
            ],
        "task_type": "classification3d",
        "src_dataset": "ShapeNet",
        "src_id": "04256520_cb71cb7b36dbcb6f826fc8d57346a2e4"
    },
    {
        ...
    }
]
```

</p>
</details> 

**Notes**：

1.  If you want to work with a specific subset of the LAMM dataset, you will need to download both the corresponding meta file and the image collection. 
2. if you prefer to download the data from the official website yourself, you can still organize it in the same way as we have and run it successfully. For example, during the 2D instruction tuning stage, if you only want to run the daily_dialogue_49k.json file, you can download the [COCO2017](http://images.cocodataset.org/zips/train2017.zip) dataset and organize it accordingly.


# LAMM-Framework
<!-- ![](./images/LAMM-Framework.png) -->
## Installation

Pre-requist Packages: `gcc <= 7.5.0; nvcc >= 11.1`


```bash
    conda create -n lamm python=3.10 -y
    conda activate lamm
    # Choose different version of torch according to your 
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
Install required packages
```bash
    pip install -r requirements.txt

    # Optional; For 3D experiments ONLY
    cd src/model/EPCL/third_party/pointnet2/
    python setup.py install
    cd ../../utils/
    pip install cython
    python cython_compile.py build_ext --inplace
```
Download required NLTK data
```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
```

Optional:

- for training

    - flash attention(v2)   
    Install flash attention (v2) if you are tight in GPU memory. Please refer to [flash attention's installation](https://github.com/Dao-AILab/flash-attention/tree/main#installation-and-features)
        > FlashAttention-2 currently supports Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100).

    - xformers   
    Install xformers if you are tight in GPU memory and cannot use flash attention (e.g., using Nvidia v100). Please refer to [xformers's installation](https://github.com/facebookresearch/xformers#installing-xformers)

- for inference

    - lightllm   
    Install lightllm to speed up inference and decrease the GPU memery usage to enable large batchsize.
        ```
        git clone -b multimodal  https://github.com/ModelTC/lightllm.git
        cd lightllm
        python setup.py install
        ```

## Data & Model Preparation for Training
- Data
    
    Follow [Download](https://github.com/OpenLAMM/LAMM/tree/readme#download) to download and prepare the data for 2D and 3D tasks. Put downloaded data in `./data` folder.
    ```
    ├── data
        ├── 2D_Instruct  
        ├── 3D_Instruct
    ```

- Language Model: Vicuna

    To prepare the pre-trained Vicuna model, please follow the instructions provided [Here](https://github.com/lm-sys/FastChat/tree/main#vicuna-weights). Put the downloaded model in the `./model_zoo/vicuna_ckpt` folder.

- 3D Encoder: EPCL

    Download Pre-trained EPCL model to tokenize point cloud from [Here](https://huggingface.co/openlamm/epcl_vit-L_256tokens/tree/main). Put the downloaded models in the `./ckpt` folder.


## Training
- 2D Models Training
    ```Bash
    cd src
    sh scripts/train_lamm2d.sh
    or
    sh scripts/train_lamm2d_slurm.sh       # for slurm
    ```
- 3D Models Training
    ```Bash
    cd src
    sh scripts/train_lamm3d.sh
    or
    sh scripts/train_lamm3d_slurm.sh       # for slurm
    ```
You need to dive into scripts to change data path and other hyper-parameters. 

For your reference, GPU memory consumption for different models are shown as follows
| Model Size | Sample Num/GPU | GPU Memory | 
| :----------: | :---------------------: | :------------------: |
|Vicuna_v0_7B | 1 | ~30GB |
|Vicuna_v0_7B | 2 | ~46GB |
|Vicuna_v0_13B | 1 | ~53GB |
|Vicuna_v0_13B | 2 | ~70GB |

# LAMM-Benchmark

**LAMM-Benchmark** evaluates 9 common image tasks, using a total of 11 datasets with over **62,439** samples, and 3 common point cloud tasks, by utilizing 3 datasets with over **12,788** data samples, while existing works only provide quantitative results on fine-tuning and evaluating specific datasets such as ScienceQA, and most works only conduct demonstration or user studies. 
- We are the very first attempt to establish a benchmark for MLLMs. We conducted a comprehensive benchmark to quantify the zero-shot and fine-tuning performance of existing multi-modal language models on various computer vision tasks and compare them against state-of-the-art methods of these tasks, including classification, object detection, pose estimation, visual question answering, facial classification, optical character recognition, object counting. 
- We also attempted two novel evaluation strategies designed explicitly for MLLMs. Specifically, as for text generation, we established a scoring logic based on the GPT API. As for tasks involving interactions between points and images, such as object detection and pose estimation, we proposed an object-locating evaluation method.

## Data & Model Preparation for LAMM-Benchmark
<details><summary> Benchmark Data For Evaluation</summary>

- 2D_Benchmark data

    |  Meta file name  | size  |  Image file name | size |  
    |  ----  | ----  |  ----  | ----  |  
    | [Caption_flickr30k.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Caption_flickr30k.json)  | 598K | flickr30k_images.zip | 559M |     
    | [Classification_CIFAR10.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Classification_CIFAR10.json)  | 2.6M | cifar10_images.zip  | 8.9M  |  
    | [Counting_FSC147.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Counting_FSC147.json) | 7.3M | fsc147_images.zip   |  44M |  
    | [Detection_VOC2012.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Detection_VOC2012.json) | 6.4M | voc2012_images.zip  | 196M  |  
    | [Facial_Classification_CelebA(Hair).json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Facial_Classification_CelebA(Hair).json) | 2.4M | celeba_images.zip  |  566M |  
    | [Facial_Classification_CelebA(Smile).json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Facial_Classification_CelebA(Smile).json) | 3.7M |  celeba_images.zip  |  566M |  
    | [Fine-grained_Classification_UCMerced.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Fine-grained_Classification_UCMerced.json) | 676K | ucmerced_images.zip  | 317M  |  
    | [Keypoints_Dectection_LSP.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Keypoints_Detection_LSP.json) | 3.9M |  fsc147_images.zip   |  44M |   
    | [Locating_FSC147.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Locating_FSC147.json) | 7.5M | fsc147_images.zip   |  44M |  
    | [Locating_LSP.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Locating_LSP.json) | 3.9M | lsp_images.zip  |  9.9M |  
    | [Locating_VOC2012.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/Locating_VOC2012.json) | 6.0M | voc2012_images.zip  | 196M  |  
    | [OCR_SVT.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/OCR_SVT.json) | 68K |  svt_images.zip  | 82M  |  
    | [VQA_AI2D.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/VQA_AI2D.json) | 2.1M | ai2d_images.zip  | 559M  |  
    | [VQA_SQAimage.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/2D_Benchmark/meta_file/VQA_SQAimage.json) | 3.6M |  sqaimage_images.zip  | 127M  |  

- 3D_Benchmark data 

    |  Meta file name  | size  |  Image file name  | size  |  
    |  ----  | ----  |  ----  | ----  |   
    |  [Detection_ScanNet.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/3D_Benchmark/meta_file/Detection_ScanNet.json)  | 1.7M  | scannet_pcls.zip  | 246M  | 
    |  [VG_ScanRefer.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/3D_Benchmark/meta_file/VG_ScanRefer.json)  | 3.7M  | scannet_pcls.zip  | 246M  | 
    |  [VQA_ScanQA_multiplechoice.json](https://huggingface.co/datasets/openlamm/LAMM_Dataset/blob/main/3D_Benchmark/meta_file/VQA_ScanQA_multiplechoice.json)  | 859K  | scannet_pcls.zip  | 246M  | 

</p>
</details> 

<details><summary> Dataset Structure </summary>

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
    │   └── meta_file  
    │       ├── Caption_flickr30k.json  
    │       ├── Classification_CIFAR10.json  
    │       ├── Counting_FSC147.json  
    │       ├── Detection_VOC2012.json  
    │       ├── Facial_Classification_CelebA(Hair).json  
    │       ├── Facial_Classification_CelebA(Smile).json  
    │       ├── Fine-grained_Classification_UCMerced.json  
    │       ├── Keypoints_Dectection_LSP.json  
    │       ├── Locating_FSC147.json  
    │       ├── Locating_LSP.json  
    │       ├── Locating_VOC2012.json  
    │       ├── OCR_SVT.json  
    │       ├── VQA_AI2D.json  
    │       └── VQA_SQAimage.json  
    └── 3D_Benchmark  
        ├── scannet_pcls.zip  
        └── meta_file  
            ├── Detection_ScanNet.json  
            ├── VG_ScanRefer.json  
            └── VQA_ScanQA_multiplechoice.json
</details>

<details><summary> Model Preparation </summary>

- Language Model: Vicuna

    To prepare the pre-trained Vicuna model, please follow the instructions provided [Here](https://github.com/lm-sys/FastChat/tree/main#vicuna-weights). Put the downloaded model in the `./model_zoo/vicuna_ckpt` folder.

- 3D Encoder: EPCL

    Download Pre-trained EPCL model to tokenize point cloud from [Here](https://huggingface.co/openlamm/epcl_vit-L_256tokens/tree/main). Put the downloaded models in the `./model_zoo/epcl_ckpt` folder.

- LAMM Models

    Download LAMM model from [Here](https://github.com/OpenLAMM/LAMM/tree/main#lamm-models). Put the downloaded models in the `./ckpt` folder.

    Or you can train your own LAMM model by following the instructions [Here](https://github.com/OpenLAMM/LAMM/tree/main#Training)!

- Other Models
    - [LLaVA](https://github.com/haotian-liu/LLaVA)
    - [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
    - [mPLUG-owl](https://github.com/X-PLUG/mPLUG-Owl)
</details>


## Evaluation 

- <details><summary>Inference trained models on 2D tasks</summary>

    ```Bash
    cd src
    sh scripts/inference_2D.sh
    ```
    or
    ``` Bash
    sh scripts/inference_2D_slurm.sh       # for slurm
    ```
- <details><summary>Inference & Evaluation on 2D tasks</summary>

    ```Bash
    sh scripts/LAMM_2D_Evaluation.sh
    ```

    or 
    ```Bash
    sh scripts/LAMM_2D_Evaluation_slurm.sh  # for slurm
    ```
    </details>
 
- <details><summary> Inference trained models on 3D tasks</summary>

    ```Bash
    cd src
    sh scripts/inference_3D.sh
    ```
    or
    ``` Bash
    sh scripts/inference_3D_slurm.sh       # for slurm
    ```
    </details>
- <details><summary> Inference & evaluation trained models on 3D tasks</summary>

    ```Bash
    sh scripts/LAMM_3D_Evaluation.sh
    ```
    or 
    ```Bash
    sh scripts/LAMM_3D_Evaluation_slurm.sh  # for slurm
    ```
    </details>

- You can also change options as following to decrease calculation consumption.
```
    --cfg ./config/train_ds3.yaml   # enable Deepspeed ZeRO stage3
    --use_flash_attn    # enable flash attention
    --use_xformers      # enable xformers
```

- Evaluation for other MLLM models. 

    Please refer to [LLaVA](https://github.com/haotian-liu/LLaVA), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and [mPLUG-owl](https://github.com/X-PLUG/mPLUG-Owl) for inference respectively. Save the answers in `./answers`. And then run `common_eval_2d.py` for evaluation. For example, to evaluate LLaVA on VOC2012:

    ```Bash
    python common_eval_2d.py \
    --dataset-name VOC2012 \
    --answer-file ./answers/LLaVA \
    --base-data-path ./data/LAMM-Dataset/2D_Benchmark \
    2>&1 | tee ./results/LLaVA/eval_VOC2012.log
    ```

- GPT Metric

    Make sure that you have finished the inference of all the evaluation dataset for both your model/LAMM model and the MLLM model to compare. For example, to rank LAMM and LLaVA: 
    ```Bash
    sh scripts/GPT_metric.sh
    ```
You may need to dive into scripts to change datasets to evaluation & checkpoints folder to load.

## Leaderboard



<details><summary> Results of LAMM model on selected 2D vision tasks </summary>
<p>

| Task                       | Dataset  | LAMM(Zero-Shot) | LAMM(Finetune) |
| -------------------------- | -------- | --------------- | -------------- |
| Classification **(Acc)**   | CIFAR10  | 37.90            | 91.2           |
| Object Detection **(Acc)** | VOC2012  | 7.20            | 13.48          |
| VQA **(mAP@0.5)**          | SQAimage | 49.88           | 74.27          |
</p>
</details>

<details><summary> Results of 3D tasks by LAMM </summary>
<p>

| Task                                         | Dataset   | SOTA  | LAMM (Zero-Shot) | LAMM (Finetune) |
| -------------------------------------------- | --------- | ----- | ---------------- | --------------- |
| 3D Object Detection **(mAP@0.5)**            | ScanNet   | 63.2  | 8.2              | 11.89           |
| Visual Grounding **(mAP@0.5)**               | ScanRefer | 54.59 | Failed           | 3.38            |
| 3D VQA **(Acc of multiple choice prolblem)** | ScanQA    | N/A   | 24.90            | 99.89           |
</p>
</details>

<details><summary> Comparison of results of Binary Locating Metric and GPT Metric of existing MLLMs </summary>
<p>

|                   | [LLaVA](https://github.com/haotian-liu/LLaVA) | [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) | [mPLUG-owl](https://github.com/X-PLUG/mPLUG-Owl) | LAMM            |
| ----------------- | ----- | -------- | --------- | --------------- |
| Binary-Loc Metric | 14.73 | 13.12    | 4.42      | **<u>36.53</u>** |
| GPT Metric        | 11    | -        | -         | **<u>89</u>**   |
</p>
</details>


<details><summary> Comparison of Multimodal Large Language Models on 2D computer vision tasks.</summary>
<p>
 Bold fonts for the best results.

| Task                  | Dataset                         | Metric     | SOTA           | [LLaVA](https://github.com/haotian-liu/LLaVA)                        | [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)                    | [mPLUG-owl](https://github.com/X-PLUG/mPLUG-Owl)         | LAMM                                  |
| --------------------- | ------------------------------- | ---------- | -------------- | ---------------------------- | --------------------------- | ----------------- | ------------------------------------- |
| Classification        | CIFAR10                         | Acc ↑      | 99.5           | **60.83**                    | 46.22                       | 42.5              | 37.9                                  |
| Detection             | VOC2012                         | mAP ↑      | 97.2           | 1.42                         | 0.92                        | 0.158             | **<u>7.20</u>**                       |
| VQA                   | SQAimage<br />AI2D              | Acc ↑      | 92.53<br />N/A | 40.5<br />18.13              | 43.43<br />Failed           | 36.39<br />19.31  | **<u>49.88</u>**<br />**<u>20.92</u>** |
| Image Caption         | flickr30k                       | BLEU4 ↑    | 30.1           | **<u>6.65</u>**              | 5.1                         | 2.74              | 2.56                                  |
| F-g clasification     | UCMerced                        | Acc ↑      | 100            | **<u>47</u>**                | 33.6                        | 32.5              | 18.23                                    |
| Counting              | FSC147                          | MAE ↓      | 10.79          | 56.2                         | Failed                      | 60.67             | **<u>46.88</u>**                      |
| OCR                   | SVT                             | Word Acc ↑ | 97.9           | **<u>37.78</u>**             | 16.97                       | 30.39             | 29.14                                   |
| Facial Classification | CelebA(Smile)<br />CelebA(Hair) | Acc ↑      | N/A<br />N/A   | Failed<br />46.42 | **<u>66.36</u>**<br />43.47 | Failed<br />40.93 | 57.60<br /> **<u>56.96</u>**                       |
| Keypoints Detection   | LSP                             | PCK ↑      | 99.5           | Failed                       | Failed                      | Failed            | Failed                                |
</p>
</details>


# LAMM Model Zoo

| # Training Samples  | Vision Encoder | LLM | Training Data | Lora Rank | Link |
| -------------------------- | :--------: | :--------: | -------- | :----: | :---------------: |
| 98K  | CLIP-ViT-L | Vicuna_v0_7B            | LAMM-2D daily dialogue & desctiption | 32 | [Checkpoints](https://huggingface.co/openlamm/lamm_7b_lora32_98k) |
| 186K  | CLIP-ViT-L | Vicuna_v0_7B            | LAMM-2D Instruction Data | 32 | [Checkpoints](https://huggingface.co/openlamm/lamm_7b_lora32_186k) |
| 186K | CLIP-ViT-L |  LLaMA2_chat_7B           | LAMM-2D Instruction Data | 32 | [Checkpoints](https://huggingface.co/openlamm/lamm186k_llama2chat7b_lora32) |
| 98K | CLIP-ViT-L | Vicuna_v0_13B           | LAMM-2D daily dialogue & desctiption | 32 | [Checkpoints](https://huggingface.co/openlamm/lamm_13b_lora32_98k) |
| 186K | CLIP-ViT-L |  Vicuna_v0_13B           | LAMM-2D Instruction Data | 32 | [Checkpoints](https://huggingface.co/openlamm/lamm_13b_lora_186k) |
| 10K | [EPCL-ViT-L](https://huggingface.co/openlamm/epcl_vit-L_256tokens/tree/main) |  Vicuna_v0_13B           | LAMM-3D Instruction Data | 32 | [Checkpoints](https://huggingface.co/openlamm/lamm3d_13b_lora32_10k) |



---

## Citation

```
    @article{yin2023lamm,
        title={LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark},
        author={Yin, Zhenfei and Wang, Jiong and Cao, Jianjian and Shi, Zhelun and Liu, Dingning and Li, Mukai and Sheng, Lu and Bai, Lei and Huang, Xiaoshui and Wang, Zhiyong and others},
        journal={arXiv preprint arXiv:2306.06687},
        year={2023}
}
```

---

## License
The project is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. The checkpoints are also CC BY NC 4.0 (allowing only non-commercial use).

---
## Acknowledgement
We thank [Hongxing Fan](https://scholar.google.com/citations?user=Wnk95ccAAAAJ), [Zeren Chen](https://github.com/Zx55), Zhen Wang for support of LAMM project. 

We also thanks the great works including [CLIP](https://github.com/openai/CLIP), [EPCL](https://arxiv.org/abs/2212.04098), [LLaMA](https://github.com/facebookresearch/llama), [Vicuna](https://github.com/lm-sys/FastChat), [FlashAttention](https://github.com/Dao-AILab/flash-attention/), [xformers](https://github.com/facebookresearch/xformers), [lightllm](https://github.com/ModelTC/lightllm)
