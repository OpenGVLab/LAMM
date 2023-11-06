<br/>

<div align="center" style={{fontSize: '50px'}}>
    <b>Octavius: Mitigating Task Interference in MLLMs via MoE</b> <br/>
</div>


<div align="center">
    Zeren Chen<sup>1,2*</sup>&emsp;
    Ziqin Wang<sup>1,3*</sup>&emsp;
    Zhen Wang<sup>2*</sup>&emsp;
    Huayang Liu<sup>2</sup>
    <br/>
    Zhenfei Yin<sup>1,4</sup>&emsp;
    Si Liu<sup>3</sup>&emsp;
    Lu Sheng<sup>2†</sup>&emsp;
    Wanli Ouyang<sup>1,4</sup>&emsp;
    Yu Qiao<sup>1</sup>&emsp;
    Jing Shao<sup>1</sup>
</div>

<div align="center">
    <sup>1</sup>Shanghai AI Laboratory&emsp;
    <sup>2</sup>School of Software, Beihang University
    <br/>
    <sup>3</sup>Institute of Artifical Intelligence, Beihang University&emsp;
    <sup>4</sup>University of Sydney
    <br/>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>†</sup> Corresponding Author
</div>

<PaperListButton 
    arxiv_link="https://www.baidu.com"
    code_link="https://www.google.com"
    model_zoo_link="/tutorial/training#octavius-model-zoo"
/>

## Introduction

We propose **Octavius**, a unified, multimodal large language with a novel capability to comprehend various tasks across different modalities, including but not limited to 2D captioning, 2D detection, 3D VQA, and 3D dense captioning. Through combining well-known Mixture-of-Experts (MoE) and one of the representative PEFT techniques, *i.e.*, LoRA, Octavius can efficiently be involved in more downstream tasks and more modalities by learning more LoRA modules, alleviating the potential task interference issuse arise from multimodal learning.

<img src="../images/Octavius_arch.png"/>
<br/>

## Usage

1. Environment [installation](https://openlamm.github.io/tutorial/installation#training).

2. Prepare the [instruction](https://openlamm.github.io/tutorial/datasets/instruction) / [benchmark](https://openlamm.github.io/tutorial/datasets/benchmark) dataset and required [pretrained weights](https://openlamm.github.io/tutorial/training#prepare-required-checkpoints) for LLMs and visual encoder.

3. Training scripts:

    - Image modality only

        ```bash
        cd src
        sh tools/Octavius/train_octavius_slurm.sh <YOUR_PARTITION>  <NUM_GPU> \
            config/Octavius/octavius_2d_e4_bs64.yaml octavius_2d_e4_bs64
        ```

    - Point cloud modality only

        ```bash
        cd src
        sh tools/Octavius/train_octavius_slurm.sh <YOUR_PARTITION>  <NUM_GPU> \
            config/Octavius/octavius_3d_e3_bs64.yaml octavius_3d_e3_bs64
        ```

    - Image & point cloud modality joint

        ```bash
        cd src
        sh tools/Octavius/train_octavius_slurm.sh <YOUR_PARTITION>  <NUM_GPU> \
            config/Octavius/octavius_2d+3d_e6_bs64.yaml octavius_2d +3d_e6_bs64
        ```

    We provide pretrained Octavius model [here](https://openlamm.github.io/tutorial/training#octavius-model-zoo).

4. Evaluation

    We use ChEF to evaluate Octavius on both image and point cloud modalities, see [here](https://openlamm.github.io/tutorial/benchmark/default#lamm-benchmark) for details.

## Citation

```bibtex
TODO
```

## License

The project is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. 