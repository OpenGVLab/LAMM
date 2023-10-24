# Tutorial

## Evaluator
In ChEF, all evaluation pipelines are managed by the [`Evaluator`](../tools/evaluator.py) class. This class serves as the control center for evaluation tasks and incorporates various components, including a scenario, an instruction, an inferencer, and a metric. These components are defined through recipe configurations.

### Key Components

- **Scenario**: The scenario represents the evaluation dataset and task-specific details.
- **Instruction**: Responsible for processing samples and generating queries.
- **Inferencer**: Performs model inference on the dataset.
- **Metric**: Evaluates model performance using defined metrics.

### Evaluation Workflow

The evaluation process in ChEF follows a structured workflow:

1. **Model and Data Loading**: First, the model and evaluation dataset (scenario) are loaded. 

2. **`Evaluator.evaluate` Method**: The evaluation is initiated by calling the `evaluate` method of the `Evaluator` class.

3. **Inference with `inferencer.inference`**: The `inferencer` is used to perform model inference. During dataset traversal, the `InstructionHandler` processes each sample, generating queries that serve as inputs to the model.

4. **Results Saving**: The output of the inference is saved in the specified `results_path`.

5. **Metric Evaluation**: Finally, the `metric` evaluates the results file, calculating various performance metrics specific to the evaluation task.

6. **Output Evaluation Results**: The final evaluation results are provided as output, allowing you to assess the model's performance.






## Employ Your Model

In ChEF, you can employ your own custom models by following these steps. This guide will walk you through the process of integrating your model into ChEF.

### Step 1: Prepare Your Model Files

1.1. Navigate to the [models](../models/) folder in ChEF.

1.2. Paste all the necessary files for your custom model into this folder.

### Step 2: Write the Test Model

2.1. Create a new Python file in [models](../models/) folder and name it something like `test_your_model.py`.

2.2. In this file, you will need to inherit from the `TestBase` class defined in [test_base.py](../models/test_base.py). The `TestBase` class provides a set of interfaces that you should implement for testing your model. 

### Step 3: Test Your Model
3.1. Add your model in [`__init__.py`](../models/__init__.py)

3.2 Prepare your model configuration in [configs/models](../configs/models/)

3.3 Use the provided recipes for evaluation. 
```shell
python tools/eval.py configs/evaluation.yaml
```

## Instruction

In ChEF, the [`InstructionHandler`](../instruction/__init__.py) class plays a central role in managing instructions for generating queries when iterating through the dataset in the `inferencer`. These queries are then used as inputs to the model for various tasks. 

ChEF supports three main query types: `standard query`, `query pool`, and `multiturn query`. For each query type, various query statements are defined based on the dataset's task type. 

1. **Standard Query**: Standard Query uses the first query defined in the query pool.
2. **Query Pool**: Query Pool specifies queries in the pool by assigned ids defined in configuration.
3. **Multiturn Query**: Multiturn Query can get different queries depending on the turn id, which are also defined in the query pool

For more details, refer to the [`query.py`](../instruction/query.py).

`InstructionHandler` also supports generate in-context examples for query, using [ice_retriever](../instruction/ice_retriever/). ChEF supports four types of ice_retrievers: `random`, `fixed`, `topk_text`, and `topk_img`. The `generate_ices` function in `InstructionHandler` class outputs several in-context examples for input query.

### Employ Your Instruction

You can add special queries in `Query Pool`, and define the assigned ids in recipe configuration to use the new queries. You can also define a new type of query by defining the query in [query.py](../instruction/query.py) and adding a new function in `InstructionHandler`.

## Inferencer

In ChEF, the `Inferencer` component is a crucial part of the system, responsible for model inference. ChEF offers a variety of pre-defined inferencers to cater to different needs. You can easily choose the appropriate inferencer by specifying the inferencer category and necessary settings in the recipe configuration. Additionally, users have the flexibility to define their custom inferencers.

### Pre-Defined Inferencers

ChEF provides eight different inferencers that cover a range of use cases. You can effortlessly use the desired inferencer by specifying its category and required settings in the recipe configuration.

### Custom Inferencers

For advanced users and specific requirements, ChEF offers the option to create custom inferencers. The basic structure of an inferencer is defined in the [`Direct.py`](../inferencer/Direct.py) file (`Direct_inferencer`). You can extend this structure to implement your custom inferencer logic.
```python
class Your_inferencer(Direct_inferencer)
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def inference(self, model, dataset):
        predictions = []
        # Step 1: build dataloader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
        for batch in tqdm(dataloader, desc="Running inference"):
            # Step 2: get input query
            prompts = self.instruction_handler.generate(batch)
            # Step 3: model outputs
            outputs = model.generate(prompts)
            # Step 4: save resuts
            predictions = predictions + outputs
        # Step 5: output file
        self._after_inference_step(predictions)
```

## Metric

In ChEF, the `Metric` component plays a crucial role in evaluating and measuring the performance of models across various scenarios and protocols. ChEF offers a wide range of pre-defined metrics, each tailored to different evaluation needs. Detailed information about these metrics can be found in the [`__init__.py`](../metric/__init__.py) file.

### Custom Metrics
ChEF also allows users to define their custom metrics. The basic structure of a metric is defined in the [`utils.py`](../metric/utils.py) file (`Base_Metric`). You can extend this structure to implement your custom metric logic.
```python
class Your_metric(Base_metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def metric_func(self, answers):
        '''
            answers: List[sample], each sample is a dict
            sample: {
                'answer' : str,
                'gt_answers' : str, 
            }
        '''
        # Evaluatation
```