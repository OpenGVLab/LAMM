# Desiderata

ChEF sets up several new evaluations to quantify the **desiderata** (desired capabilities) that a competent MLLM model should possess, as a reliable agent that can perform real-world multimodal interactions.

- [calibration](#calibration)
- [in-context learning](#in-context-learning)
- [instruction following](#instruction-following)
- [language performance](#language-performance)
- [robustness](#robustness)
- [hallucination](#hallucination)


## Calibration
Calibration evaluates how the uncertainty about each MLLM’s prediction is aligned with its accuracy, as highlighted by HELM. ChEF provides the calibration evaluation on [MMBench](../configs/desiderata_recipes/Calibration/MMBench.yaml) and [ScienceQA](../configs/desiderata_recipes/Calibration/ScienceQA.yaml).

```shell
python tools/desiderata/eval_calibration.py configs/desiderata_recipes/Calibration/evaluation.yaml
```

## In-context Learning
In-context learning evaluates the crucial in-context learning (ICL) ability of an MLLM. ChEF provides the in-context learning evaluation on [MMBench](../configs/desiderata_recipes/ICL/MMBench.yaml) and [ScienceQA](../configs/desiderata_recipes/ICL/ScienceQA.yaml).
```shell
python tools/desiderata/eval_icl.py configs/desiderata_recipes/ICL/evaluation.yaml
```

## Instruction Following
Instruction following evaluates how exactly the MLLM relies on the given instructions. ChEF provides the instruction following evaluation on [MMBench](../configs/desiderata_recipes/Insfollow/MMBench.yaml) and [ScienceQA](../configs/desiderata_recipes/Insfollow/ScienceQA.yaml).
```shell
python tools/desiderata/eval_insfollow.py configs/desiderata_recipes/Insfollow/evaluation.yaml
```

## Language Performance
Language performance evaluates the quality of the generated sentences. ChEF uses the GPT-based metric. Before evaluate the language performance, please first finish the inference on MMBench and ScienceQA, using the default recipes. [MMBench_recipe](../configs/scenario_recipes/MMBench/default.yaml), [ScienceQA_recipe](../configs/scenario_recipes/ScienceQA/default.yaml)
```shell
python tools/desiderata/eval_langperf.py --base-data-path dataset_path --answer-path results_path --response-dir output_path
```
## Robustness
Robustness measures how robust an MLLM is to corruption in the multimodal inputs. ChEF provides the robustness evaluation on [MMBench](../configs/desiderata_recipes/Robust/MMBench.yaml) and [ScienceQA](../configs/desiderata_recipes/Robust/ScienceQA.yaml).
```shell
python tools/desiderata/eval_robust.py configs/desiderata_recipes/Robust/evaluation.yaml
```

## Hallucination
Hallucination evaluates how an MLLM avoids mentioning visual objects that do not exist in the images. ChEF uses [POPE](../configs/desiderata_recipes/Hallucination) for hallucination evaluation. 
```shell
python tools/desiderata/eval_hallucination.py
```


## More Desiderata

New desideratum can be implemented by setting new recipes. See [tutorial.md](tutorial.md) for details.
