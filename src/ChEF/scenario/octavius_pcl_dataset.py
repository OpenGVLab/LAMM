from collections import defaultdict
import json
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .lamm_sysmsg import common_task2sysmsg


class OctaviusPCLDataset(Dataset):

    def __init__(self, base_data_path: str, task_name: str, inference_dataset_name: str, vision_root_path: str, **kwargs):
        super().__init__()
        self.task_name = task_name + '_octavius3d'
        self.dataset_name = inference_dataset_name + '_' + task_name
        self.data = []
        self.system_msg = common_task2sysmsg[task_name + '3D']

        data_file_path = os.path.join(base_data_path, task_name + '_' + inference_dataset_name + '.json')
        with open(data_file_path, 'r') as f:
            json_data = json.load(f)
        pickle_root = '/'.join(data_file_path.split('/')[:-1])
        pickle_path = f'{pickle_root}/{task_name}_{inference_dataset_name}.pickle'
        
        if 'scannet' in inference_dataset_name or 'nr3d' in inference_dataset_name:
            pc_scene_dataset = 'lamm_scannet_tr3d'
            pc_obj_dataset = 'lamm_scannet_gt'
        elif 'shapenet' in inference_dataset_name:
            pc_scene_dataset = pc_obj_dataset = 'shapenet_pcls'
        
        self.vision_embeds_3d_ref_list, self.vision_embeds_3d_scene_prop_list = [], []
        self.vision_pos_3d_ref_list, self.vision_pos_3d_scene_prop_list = [], []
        self.caption_list, self.task_type_list = [], []
        self.scene_id_list = []
        self.object_name_ilst = []
        
        self.max_proposal_num = 0
        scene_id_to_3d_embeds = {}
        scene_id_to_3d_pos = {}
        
        for item in tqdm(json_data, desc='loading 3d data'):
            self.object_name_ilst.append(item['object_name'])
            
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            self.vision_embeds_3d_ref_list = data['vision_embeds_3d_ref_list']
            self.vision_embeds_3d_scene_prop_list = data['vision_embeds_3d_scene_prop_list']
            self.vision_pos_3d_ref_list = data['vision_pos_3d_ref_list']
            self.vision_pos_3d_scene_prop_list = data['vision_pos_3d_scene_prop_list']
            self.caption_list = data['caption_list']
            self.task_type_list = data['task_type_list']
            self.max_proposal_num = data['max_proposal_num']
            self.scene_id_list = data['scene_id_list'] if 'scene_id_list' in data else []
        else:
            for scene_id in tqdm(os.listdir(os.path.join(vision_root_path, pc_scene_dataset, 'ins_pc_feat')), desc="generate scene features"):
                # scene-level 3d prop vision embeds
                scene_prop_feat_3d_root = os.path.join(vision_root_path, pc_scene_dataset, 'ins_pc_feat', scene_id)
                obj_prop_path_list = sorted(os.listdir(scene_prop_feat_3d_root))
                self.max_proposal_num = max(self.max_proposal_num, len(obj_prop_path_list))
                scene_gt_3d_feat = []
                for obj_prop_path in obj_prop_path_list:
                    scene_gt_3d_feat.append(torch.tensor(np.load(os.path.join(scene_prop_feat_3d_root, obj_prop_path)), dtype=torch.float16)) 
                scene_id_to_3d_embeds[scene_id] = torch.stack(scene_gt_3d_feat)
                
                # scene-level 3d prop pos, we need to convert (8, 3) to center+size (6,)
                scene_prop_pos_3d_root = os.path.join(vision_root_path, pc_scene_dataset, 'bbox', scene_id)
                scene_prop_pos_3d = []
                for obj_prop_path in obj_prop_path_list:
                    obj_prop_path = obj_prop_path.split('.')[0].split('-')
                    obj_prop_id = obj_prop_path[0]
                    obj_prop_name = '-'.join(obj_prop_path[1:])
                    obj_prop_bbox = np.load(os.path.join(scene_prop_pos_3d_root, f'{obj_prop_id}-{obj_prop_name}.npy'))
                    scene_prop_pos_3d.append(torch.tensor(np.concatenate([obj_prop_bbox.min(axis=0), obj_prop_bbox.max(axis=0)]), dtype=torch.float16))
                
                scene_id_to_3d_pos[scene_id] = torch.stack(scene_prop_pos_3d)
                
            for item in tqdm(json_data, desc='loading 3d training data'):
                task_type, caption = item.get('task_type', 'normal'), item['conversations']
                self.caption_list.append(caption)
                self.task_type_list.append(task_type)
                scene_id = item['scene_id']
                self.scene_id_list.append(scene_id)
                
                ###############################
                # Deal with Scene level Input #
                ###############################
                
                self.vision_embeds_3d_scene_prop_list.append(scene_id_to_3d_embeds[item['scene_id']])
                self.vision_pos_3d_scene_prop_list.append(scene_id_to_3d_pos[item['scene_id']])
                
                #############################
                # Deal with Obj level Input #
                #############################
                
                # reference object info, vqa task has no reference object
                if task_type == 'VQA3D':
                    ref_obj_name = ref_obj_id = None
                    self.vision_embeds_3d_ref_list.append(torch.tensor(np.zeros(768), dtype=torch.float16))
                    self.vision_pos_3d_ref_list.append(torch.tensor(np.zeros(6), dtype=torch.float16))
                else:
                    ref_obj_name = '_'.join(item['object_name'].split(' '))
                    ref_obj_id = item['object_id']
                
                    # obj-level 3d prop vision embeds
                    vision_embeds_3d_ref = torch.tensor(np.load(os.path.join(vision_root_path, pc_obj_dataset, 'ins_pc_feat', scene_id, f'{ref_obj_id}-{ref_obj_name}.npy')), dtype=torch.float16)
                    self.vision_embeds_3d_ref_list.append(vision_embeds_3d_ref.reshape(-1))
                    
                    # obj-level 3d prop pos, we need to convert (8, 3) to center+size (6,)
                    vision_pos_3d_ref = np.load(os.path.join(vision_root_path, pc_obj_dataset, 'bbox', scene_id, f'{ref_obj_id}-{ref_obj_name}.npy'))
                    vision_pos_3d_ref = torch.tensor(np.concatenate([vision_pos_3d_ref.min(axis=0), vision_pos_3d_ref.max(axis=0)]), dtype=torch.float16)
                    self.vision_pos_3d_ref_list.append(vision_pos_3d_ref.reshape(-1))
            
            data = {}
            data['caption_list'] = self.caption_list
            data['task_type_list'] = self.task_type_list
            data['vision_embeds_3d_ref_list'] = self.vision_embeds_3d_ref_list
            data['vision_embeds_3d_scene_prop_list'] = self.vision_embeds_3d_scene_prop_list
            data['vision_pos_3d_ref_list'] = self.vision_pos_3d_ref_list
            data['vision_pos_3d_scene_prop_list'] = self.vision_pos_3d_scene_prop_list
            data['max_proposal_num'] = self.max_proposal_num
            data['scene_id_list'] = self.scene_id_list
            data['object_name_ilst'] = self.object_name_ilst
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)

    def __len__(self):
        """get dataset length
        :return int: length of dataset
        """
        return len(self.task_type_list)
    
    def __getitem__(self, i):
        """get one sample"""
        output_texts=self.caption_list[i]
        task_type=self.task_type_list[i]
        vision_embeds_3d_ref=self.vision_embeds_3d_ref_list[i]
        vision_embeds_3d_scene_prop=self.vision_embeds_3d_scene_prop_list[i]
        vision_pos_3d_ref=self.vision_pos_3d_ref_list[i]
        vision_pos_3d_scene_prop=self.vision_pos_3d_scene_prop_list[i]
        scene_id = self.scene_id_list[i] if len(self.scene_id_list) > 0 else None
        object_name = self.object_name_ilst[i] if len(self.object_name_ilst) > 0 else None
        
        vision_embeds_3d_scene_prop_padding = torch.zeros(self.max_proposal_num, vision_embeds_3d_scene_prop.shape[-1])
        vision_embeds_3d_scene_prop_padding[:vision_embeds_3d_scene_prop.shape[0]] = vision_embeds_3d_scene_prop
        
        vision_pos_3d_scene_prop_padding = torch.zeros(self.max_proposal_num, vision_pos_3d_scene_prop.shape[-1])
        vision_pos_3d_scene_prop_padding[:vision_embeds_3d_scene_prop.shape[0]] = vision_pos_3d_scene_prop
        
        mask = torch.zeros(self.max_proposal_num)
        mask[:vision_embeds_3d_scene_prop.shape[0]] = 1
        return dict(
            question=output_texts[0]['value'],
            gt=output_texts[1]['value'],
            task_type=task_type,
            vision_embeds_3d_ref=vision_embeds_3d_ref.reshape(-1),
            vision_embeds_3d_scene_prop=vision_embeds_3d_scene_prop_padding,
            vision_pos_3d_ref=vision_pos_3d_ref.reshape(-1),
            vision_pos_3d_scene_prop=vision_pos_3d_scene_prop_padding,
            mask=mask,
            scene_id=scene_id,
            object_name=object_name,
        )

    def collate(self, instances):
        """collate function for dataloader"""
        keys = [key for key in instances[0].keys()]
        return_dict = defaultdict()
        for key in keys:
            return_dict[key] = []
            for instance in instances:
                return_dict[key].append(instance[key])
            if isinstance(instance[key], torch.Tensor):
                if key=='scene_scale':
                    return_dict[key] = torch.stack(return_dict[key])
                else:
                    return_dict[key] = torch.stack(return_dict[key]).half()
            
        return return_dict
