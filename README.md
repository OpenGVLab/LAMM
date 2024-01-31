# LAMM

LAMM (pronounced as /læm/, means cute lamb to show appreciation to LLaMA), is a growing open-source community aimed at helping researchers and developers quickly train and evaluate Multi-modal Large Language Models (MLLM), and further build multi-modal AI agents capable of bridging the gap between ideas and execution, enabling seamless interaction between humans and AI machines.

<p align="center">
    <font size='4'>
    <a href="https://openlamm.github.io/" target="_blank">🌏 Project Page</a>
    </font>
</p>

## Updates
📆 [**2023-12**] 
1. [DepictQA](https://arxiv.org/abs/2312.08962): Depicted Image Quality Assessment based on Multi-modal Language Models released on Arxiv!
2. [MP5](https://arxiv.org/abs/2312.07472): A Multi-modal LLM based Open-ended Embodied System in Minecraft released on Arxiv!

📆 [**2023-11**] 

1. [ChEF](https://openlamm.github.io/paper_list/ChEF): A comprehensive evaluation framework for MLLM released on Arxiv!
2. [Octavius](https://openlamm.github.io/paper_list/Octavius): Mitigating Task Interference in MLLMs by combining Mixture-of-Experts (MoEs) with LoRAs released on Arxiv!
3. Camera ready version of LAMM is available on [Arxiv](https://arxiv.org/abs/2306.06687).

📆 [**2023-10**]
1. LAMM is accepted by NeurIPS2023 Datasets & Benchmark Track! See you in December!

📆 [**2023-09**]
1. Light training framework for V100 or RTX3090 is available! LLaMA2-based finetuning is also online.
2. Our demo moved to <a href="https://openxlab.org.cn/apps/detail/LAMM/LAMM" target="_blank">OpenXLab</a>.

📆 [**2023-07**]
1.  Checkpoints & Leaderboard of LAMM on huggingface updated on new code base.
2.  Evaluation code for both 2D and 3D tasks are ready.
3.  Command line demo tools updated.

📆 [**2023-06**]
1. LAMM: 2D & 3D dataset & benchmark for MLLM
2. Watch demo video for LAMM at <a href="https://www.youtube.com/watch?v=M7XlIe8hhPk" target="_blank">YouTube</a> or <a href="https://www.bilibili.com/video/BV1kN411D7kt/" target="_blank">Bilibili</a>!
3. Full paper with Appendix is available on <a href="https://arxiv.org/abs/2306.06687" target="_blank">Arxiv</a>.
4. LAMM dataset released on <a href="https://huggingface.co/datasets/openlamm/LAMM_Dataset" target="_blank">Huggingface</a> & <a href="https://opendatalab.com/LAMM/LAMM" target="_blank">OpenDataLab</a> for Research community!',
5. LAMM code is available for Research community!


## Paper List
**Publications**

- [x] [LAMM](https://openlamm.github.io/paper_list/LAMM)
- [x] [Octavius](https://openlamm.github.io/paper_list/Octavius)


**Preprints**
- [x] [ChEF](https://openlamm.github.io/paper_list/ChEF)

## Citation
**LAMM**

```
@article{yin2023lamm,
    title={LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark},
    author={Yin, Zhenfei and Wang, Jiong and Cao, Jianjian and Shi, Zhelun and Liu, Dingning and Li, Mukai and Sheng, Lu and Bai, Lei and Huang, Xiaoshui and Wang, Zhiyong and others},
    journal={arXiv preprint arXiv:2306.06687},
    year={2023}
}
```

**ChEF**

```
@misc{shi2023chef,
      title={ChEF: A Comprehensive Evaluation Framework for Standardized Assessment of Multimodal Large Language Models}, 
      author={Zhelun Shi and Zhipin Wang and Hongxing Fan and Zhenfei Yin and Lu Sheng and Yu Qiao and Jing Shao},
      year={2023},
      eprint={2311.02692},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**Octavius**

```
@misc{chen2023octavius,
      title={Octavius: Mitigating Task Interference in MLLMs via MoE}, 
      author={Zeren Chen and Ziqin Wang and Zhen Wang and Huayang Liu and Zhenfei Yin and Si Liu and Lu Sheng and Wanli Ouyang and Yu Qiao and Jing Shao},
      year={2023},
      eprint={2311.02684},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**DepictQA**

```
@article{depictqa,
        title={Depicting Beyond Scores: Advancing Image Quality Assessment through Multi-modal Language Models},
        author={You, Zhiyuan and Li, Zheyuan, and Gu, Jinjin, and Yin, Zhenfei and Xue, Tianfan and Dong, Chao},
        journal={arXiv preprint arXiv:2312.08962},
        year={2023}
    }
```

**MP5**

```
@misc{qin2023mp5,
  title         = {MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception}, 
  author        = {Yiran Qin and Enshen Zhou and Qichang Liu and Zhenfei Yin and Lu Sheng and Ruimao Zhang and Yu Qiao and Jing Shao},
  year          = {2023},
  eprint        = {2312.07472},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```


## Get Started
Please see [tutorial](https://openlamm.github.io/tutorial) for the basic usage of this repo.

## License 

The project is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.