# Get Started

## Overall Architecture

```
LAMM
└── ckpt
│   ├── epcl_vit-L_256tokens  # EPCL pretraining checkpoints (Optional)
│   ├── lamm_2d               # saved checkpoints in training
│   └── ...
└── data                      # dataset folder, see `Dataset Preparation` section for detail
│   ├── LAMM                  # LAMM dataset
│   ├── Octavius              # Octavius dataset
│   ├── ChEF                  # ChEF dataset  
│   └── ...                   # your custom dataset
└── model_zoo                 # see `Model Preparation for Training` for detail 
│   └── vicuna_ckpt
└── src
└── ...
```

## Dataset Preparation

### LAMM-Dataset

You can download LAMM-Dataset from [here](https://opendatalab.com/LAMM/download). Here is the table illustrating the correspondence between each Meta file and image collection in the LAMM dataset:

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

```
data
└── LAMM
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
    │   ├── 3rscan_pcls.zip  
    │   ├── shapenet_pcls.zip  
    │   └── meta_file  
    │       └── LAMM_3dinstruct_10k.json  
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
```

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
                "value": "The kitchen in the image is furnished with white cabinets and white   appliances. There is a dishwasher, a stove, and a sink. On the stove, a blue towel    hangs on the handle. A cutting board is placed on the dishwasher. There are also   additional elements like a bowl of apples on the counter and a beige rug on the floor."
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
                        "value": "What scenario does this point cloud belong to according to the    model\u2019s prediction?"
                    },
                    {
                        "from": "gpt",
                        "value": "Through meticulous analysis, it becomes evident that the point cloud  aligns with the characteristics of sofa,couch,lounge s       cenario."
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

### Octavius

You can download LAMM-Dataset from [here](https://opendatalab.com/LAMM/OctaviusDataset). 

For 2D instruction dataset, we use the same dataset collection as LAMM (coco/bamboo/locount/textvqa), and provide corresponding meta file [here](https://opendatalab.com/LAMM/OctaviusDataset/tree/main/OctaviusDataset_2D/meta_file). 2D benchmark dataset is same as LAMM.

For 3D instruction dataset,

<details><summary> Dataset Structure </summary>

```
data
└── Octavius
    └── 2D_Instruct  
    │   ├── coco_images.zip  
    │   ├── bamboo_images.zip  
    │   ├── textvqa_images.zip  
    │   ├── locount_images.zip  
    │   └── meta_file  
    │       └── octavius_2d_train_293k.json
    └── 3D_Instruct  
    │   ├── scan2inst_train.pickle
    │   └── meta_file  
    │       └── scan2inst_train.json  
    ├── 2D_Benchmark  # same as LAMM, make a symbol link to ../LAMM/2D_Benchmark
    └── 3D_Benchmark  
        ├── Caption_nr3d.pickle
        ├── Caption_scannet.pickle
        ├── Classification_scannet.pickle
        ├── Classification_shapenet.pickle
        ├── VQA_scannet.pickle
        └── meta_file  
            ├── Caption_nr3d.json
            ├── Caption_scannet.json
            ├── Classification_scannet.json
            ├── Classification_shapenet.json
            ├── Detection_ScanNet.json
            ├── VG_ScanRefer.json
            ├── VQA_scannet.json
            └── VQA_ScanQA_multiplechoice.json
```

</p>
</details> 

### ChEF 

#### LAMM
Download LAMM 2D Benchmark datasets. More details are in [LAMM-Dataset](#lamm-dataset). 

<details><summary> Data Structure </summary> 

```text
ChEF
├── configs
└── data
    ├── checkpoints
    └── datasets
        └── LAMM
            └── 2D_Benchmark
                ├── cifar10_images
                ├── flickr30k_images
                ├── fsc147_images
                ├── meta_file
                ├── sqaimage_images
                └── voc2012_images
```
</details>

#### Omnibenchmark
Download [Omnibenchmark](https://entuedu-my.sharepoint.com/:f:/g/personal/yuanhan002_e_ntu_edu_sg/El2wmbzutJBOlu8Tz9HyDJABMmFtsG_8mq7uGh4Q7F1QSQ?e=NyroDS) for fine-grained classification dataset and [Bamboo Label System](https://github.com/ZhangYuanhan-AI/Bamboo) for hierarchical catergory labels. 

<details><summary> Data Structure </summary>

```text
ChEF
├── configs
└── data
    ├── checkpoints
    └── datasets
        ├── Omnibenchmark_raw
        └── Bamboo
            └── sensexo_visual_add_academic_add_state_V4.visual.json
```

We sampled and labeled Omnibenchmark meticulously by using a hierarchical chain of categories, facilitated by the Bamboo label system. 
```shell
python ChEF/data_process/Omnibenchmark.py
```

You can also directly download the labeled Omnibenchmark dataset from [OpenXLab](https://openxlab.org.cn/datasets/LAMM/ChEF).

Finally, the dataset should have this structure:

```text
ChEF
├── configs
└── data
    ├── checkpoints
    └── datasets
        ├── ChEF
        |   └── Omnibenchmark_Bamboo
        |       ├── meta_file
        |       └── omnibenchmark_images
        └── Bamboo
            └── sensexo_visual_add_academic_add_state_V4.visual.json
```
</details>

#### MMBench, MME and SEEDBench
Refer to [MMBench](https://github.com/open-compass/MMBench), [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) and [SEEDBench](https://github.com/AILab-CVC/SEED-Bench) for dataset and more details.

<details><summary> Data Structure </summary>

```text
ChEF
├── configs
└── data
    ├── checkpoints
    └── datasets
        ├── MMBench
        |   ├── mmbench_dev_20230712.tsv
        |   └── mmbench_test_20230712.tsv
        ├── MME_Benchmark_release_version
        └── SEED-Bench
```

</details>

#### POPE
POPE is a special labeled COCO dataset for hallucination evaluation based on the validation set of COCO 2014. Download [COCO](https://cocodataset.org/#download)  and [POPE](https://github.com/RUCAIBox/POPE).

<details><summary> Data Structure </summary>

```text
ChEF
├── configs
└── data
    ├── checkpoints
    └── datasets
        └── coco_pope
            ├── val2014
            ├── coco_pope_adversarial.json
            ├── coco_pope_popular.json
            └── coco_pope_random.json
```

</details>

#### MMBench_C and ScienceQA_C
MMBench_C and ScienceQA_C are datasets with image and text corruptions fot robustness evaluation. You can also directly download the MMBench_C and ScienceQA_C dataset from [OpenXLab](https://openxlab.org.cn/datasets/LAMM/ChEF).

<details><summary> Data Structure </summary>

```text
ChEF
├── configs
└── data
    ├── checkpoints
    └── datasets
        └── ChEF
            ├── MMBench_C
            |   ├── images
            |   ├── Image_Corruptions_info.json
            |   ├── Text_Corruptions_info.json
            |   └── MMBench_C.json
            └── ScienceQA_C
                ├── sqaimage_images
                ├── Image_Corruptions_info.json
                ├── Text_Corruptions_info.json
                └── VQA_ScienceQA_C.json
```

</details>

## Model Preparation

- Language Model: Vicuna

    To prepare the pre-trained Vicuna model, please follow the instructions provided [Here](https://github.com/lm-sys/FastChat/tree/main#vicuna-weights). Put the downloaded model in the `./model_zoo/vicuna_ckpt` folder.

- 3D Encoder: EPCL

    Download Pre-trained EPCL model to tokenize point cloud from [Here](https://huggingface.co/openlamm/epcl_vit-L_256tokens/tree/main). Put the downloaded models in the `./ckpt` folder.

- LAMM Pretrained Models

    Download LAMM model from [Here](https://github.com/OpenLAMM/LAMM/tree/main#lamm-models). Put the downloaded models in the `./ckpt` folder.

    Or you can train your own LAMM model by following the instructions [Here](https://github.com/OpenLAMM/LAMM/tree/main#Training)!

- Octavius Pretrained Models

    Download pretrained Octavius model from [Here]().

- ChEF Models

## Environment Setup

Pre-requist Packages: `gcc <= 7.5.0; nvcc >= 11.1`

### LAMM-Framework & Octavius

1. Python & Pytorch Environment

    ```bash
    conda create -n lamm python=3.10 -y
    conda activate lamm
    # Choose different version of torch according to your 
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
    ```

2. Install Required Dependencies

    ```bash
    conda install timm==0.6.7 deepspeed==0.9.3 transformers==4.31.0 -c conda-forge
    pip install peft==0.3.0 --no-dependencies
    pip install -r requirements/default.txt
    ```
    Download required NLTK data

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

3. Optional Features & Dependencies

    * LAMM-3D Environments

        ```bash
        cd src/model/EPCL/third_party/pointnet2/
        python setup.py install
        cd ../../utils/
        pip install cython
        python cython_compile.py build_ext --inplace
        ```

    * Reducing Memory in Training

        - flash attention (v2)   
            
            Install flash attention (v2) if you are tight in GPU memory. Please refer to [flash attention's installation](https://github.com/Dao-AILab/flash-attention/tree/main#installation-and-features)
            
            > FlashAttention-2 currently supports Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100).

        - xformers   
    
            Install xformers if you are tight in GPU memory and cannot use flash attention (e.g., using Nvidia v100). Please refer to [xformers's installation](https://github.com/facebookresearch/xformers#installing-xformers)

    * Efficient Inference

        - lightllm   
            
            Install lightllm to speed up inference and decrease the GPU memery usage to enable large batchsize.
        
            ```bash
            git clone -b multimodal  https://github.com/ModelTC/lightllm.git
            cd lightllm
            python setup.py install
            ```
    
### ChEF

1. Prepare the environment
    ```shell
    conda create -n ChEF python=3.10
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements/default.txt
    pip install -r requirements/ChEF.txt
    ```
2. Prepare the MLLM

    See [models.md](./ChEF/models.md) for details. 



## Training

### LAMM

- 2D Models Training
    
    ```bash
    cd src
    sh tools/LAMM/train_lamm2d.sh lamm_2d
    # or
    sh tools/LAMM/train_lamm2d_slurm.sh <YOUR_PARTITION> lamm_2d
    ```

- 3D Models Training

    ```bash
    cd src
    sh tools/LAMM/train_lamm3d.sh lamm_3d
    # or
    sh tools/LAMM/train_lamm3d_slurm.sh <YOUR_PARTITION> lamm_3d
    ```

For your reference, GPU memory consumption for different models are shown as follows

| Model Size | Sample Num/GPU | GPU Memory | 
| :----------: | :---------------------: | :------------------: |
|Vicuna_v0_7B | 1 | ~30GB |
|Vicuna_v0_7B | 2 | ~46GB |
|Vicuna_v0_13B | 1 | ~53GB |
|Vicuna_v0_13B | 2 | ~70GB |

### Octavius

- Image modality only

    ```bash
    cd src
    sh tools/Octavius/train_octavius_slurm.sh <YOUR_PARTITION> <NUM_GPU> \
        config/Octavius/octavius_2d_e4_bs64.yaml octavius_2d_e4_bs64
    ```

- Point cloud modality only

    ```bash
    cd src
    sh tools/Octavius/train_octavius_slurm.sh <YOUR_PARTITION> <NUM_GPU> \
        config/Octavius/octavius_3d_e3_bs64.yaml octavius_3d_e3_bs64
    ```

- Image & point cloud modality joint

    ```bash
    cd src
    sh tools/Octavius/train_octavius_slurm.sh <YOUR_PARTITION> <NUM_GPU> \
        config/Octavius/octavius_2d+3d_e6_bs64.yaml octavius_2d+3d_e6_bs64
    ```

## Benchmark


### LAMM-Benchmark

**Notes**: LAMM-Benchmark has now been fully implemented using ChEF, and we highly recommend using the latest ChEF evaluation method for benchmarking in your work. Please refer to [ChEF](#chef-2) for detailed information.

ChEF supports the common 2D and 3D tasks evaluation and locating tasks evaluation in LAMM. Please note that the GPT rank metric in LAMM is no longer applicable.

To evaluate LAMM on LAMM-Benchmark in 2D common tasks, use the defined [model_cfg](../src/config/ChEF/models/lamm.yaml) and the defined recipes in [lamm_scenario_recipes](../src/config/ChEF/scenario_recipes/LAMM/). For example, to evaluate LAMM on ScienceQA, run:
```shell
python eval.py --model_cfg config/ChEF/models/lamm.yaml  --recipe_cfg config/ChEF/scenario_recipes/LAMM/ScienceQA.yaml
```

To evaluate LAMM on ScanNet Detection, run:
```shell
python eval.py --model_cfg config/ChEF/models/lamm_3d.yaml  --recipe_cfg config/ChEF/scenario_recipes/LAMM/ScanNet.yaml
```
If you want to automately running all the evaluations sequentially, you can run
```shell
sh tools/LAMM/eval_lamm2d.sh
sh tools/LAMM/eval_lamm3d.sh
```

### ChEF

1. Visual performance evaluation
    We provide several recipes and model configs in ChEF/configs. Define the models and recipes in [configs](../src/config/ChEF/)

    For example, to evaluate LAMM on CIFAR10 using the default recipe, run:
    ```shell
    python tools/eval.py --model_cfg config/ChEF/models/lamm.yaml --recipe_cfg config/ChEF/scenario_recipes/CIFAR10/default.yaml
    ```
2. Desiderata
    To evaluate the desiderata, see [desiderata.md](./ChEF/desiderata.md) for details.
3. Custom evaluation
    To deploy your own model or design your own recipe, see [tutorial.md](./ChEF/tutorial.md) for details.