import json
import os
from instruction import build_instructionhandler
from inferencer import build_inferencer
from metric import build_metric

class Evaluator:
    def __init__(self,
                 dataset,
                 save_base_dir,
                 cfg,
                 **kwargs) -> None:
        self.dataset = dataset
        self.dataset_name = self.dataset.dataset_name
        self.task_name = self.dataset.task_name
        self.save_base_dir = save_base_dir
        instruction_cfg = cfg['instruction_cfg']
        self.instruction_handler = build_instructionhandler(task_name=self.task_name, dataset=self.dataset, **instruction_cfg)
        inferencer_cfg = cfg['inferencer_cfg']
        self.inferencer = build_inferencer(dataset_name = self.dataset_name,
                                           save_base_dir = save_base_dir,
                                           instruction_handler = self.instruction_handler,
                                           **inferencer_cfg)
        
        metric_cfg = cfg['metric_cfg']
        self.metric = build_metric(dataset_name=self.dataset_name, 
                                   **metric_cfg)
        
        
    def evaluate(self, model):
        model.ice_imgs_emb = None
        self.inferencer.inference(model, self.dataset)
        results_path = self.inferencer.results_path
        result = self.metric.metric(results_path)
        with open(os.path.join(self.save_base_dir, 'results.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(dict(
                        answer_path = results_path,
                        result = result
                    ), indent=4))




def build_evaluator(dataset, task_name, save_base_dir, eval_cfg, **kwargs):
    return Evaluator(dataset=dataset, save_base_dir=save_base_dir, cfg=eval_cfg)