#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import defaultdict
import copy
from dataclasses import dataclass, field
import io
import json
import numpy as np
import os
import pickle
import random
from tqdm import tqdm
import xml.dom.minidom
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import transformers
from typing import Callable, Dict, Sequence


class LAMMDataset(Dataset):
    """LAMM Dataset"""

    def __init__(self, data_file_path: str, vision_root_path: str, vision_type="image"):
        """Initialize supervised datasets

        :param str data_file_path: path of conversation file path
        :param str vision_root_path: vision root path
        :param str vision_type: type of vision data, defaults to 'image', image / pcl
        """
        super(LAMMDataset, self).__init__()
        self.vision_type = vision_type
        with open(data_file_path, "r") as fr:
            json_data = json.load(fr)

        self.vision_path_list, self.caption_list, self.task_type_list = [], [], []
        for item in json_data:
            if not vision_type in item:
                continue
            one_vision_name, one_caption = item[vision_type], item["conversations"]
            task_type = item["task_type"] if "task_type" in item else "normal"

            if not one_vision_name.startswith("/"):
                one_vision_path = os.path.join(vision_root_path, one_vision_name)
            else:
                one_vision_path = one_vision_name

            self.vision_path_list.append(one_vision_path)
            self.caption_list.append(one_caption)
            self.task_type_list.append(task_type)
        print(f"[!] collect {len(self.vision_path_list)} samples for training")

    def __len__(self):
        """get dataset length

        :return int: length of dataset
        """
        return len(self.vision_path_list)

    def __getitem__(self, i):
        """get one sample"""
        return dict(
            vision_paths=self.vision_path_list[i],
            output_texts=self.caption_list[i],
            vision_type=self.vision_type,
            task_type=self.task_type_list[i],
        )

    def collate(self, instances):
        """collate function for dataloader"""
        vision_paths, output_texts, task_type = tuple(
            [instance[key] for instance in instances]
            for key in ("vision_paths", "output_texts", "task_type")
        )
        return dict(
            vision_paths=vision_paths,
            output_texts=output_texts,
            vision_type=self.vision_type,
            task_type=task_type,
        )


def clip(a: float):
    if a > 1:
        a = 1
    if a < 0:
        a = 0
    return a

    
class OctaviusDataset(Dataset):
    def __init__(
        self, 
        data_file_path_2d: str, 
        data_file_path_3d: str,
        vision_root_path_2d: str,
        vision_root_path_3d: str,
        loop_2d: int,
        loop_3d: int,
        **kwargs
    ):
        super().__init__()
        self.vision_type = ['image', 'pcl']
        self.data_file_path_2d = data_file_path_2d
        self.data_file_path_3d = data_file_path_3d
        self.vision_root_path_2d = vision_root_path_2d
        self.vision_root_path_3d = vision_root_path_3d
        self.loop_2d = loop_2d
        self.loop_3d = loop_3d

        self.data_2d = self.prepare_2d_data()
        self.len_2d = len(self.data_2d['vision_path_list']) 
        self.data_3d = self.prepare_3d_data()
        self.len_3d = len(self.data_3d['task_type_list'])

        self.vision_type_list = ['img'] * (self.len_2d * self.loop_2d) + ['pcl'] * (self.len_3d * self.loop_3d)
        self.index_list = list(range(self.len_2d)) * self.loop_2d + list(range(self.len_3d)) * self.loop_3d

    def __len__(self):
        return self.len_2d * self.loop_2d + self.len_3d * self.loop_3d
    
    def __getitem__(self, i):
        vision_type = self.vision_type_list[i]
        index = self.index_list[i]

        if vision_type == 'img':
            return self.get_2d_data(index)
        else:
            return self.get_3d_data(index)

    def get_2d_data(self, index):
        def _get_elem_from_tree(tree, tag):
            return tree.getElementsByTagName(tag)[0].firstChild.data
        
        vision_paths = self.data_2d['vision_path_list'][index]
        output_texts = self.data_2d['caption_list'][index]
        task_type = self.data_2d['task_type_list'][index]

        if task_type == "detection":
            for i in range(len(output_texts)):
                if output_texts[i]["from"] == "gpt":
                    xml_text = output_texts[i]["value"]
                    dom_tree = xml.dom.minidom.parse(io.StringIO(xml_text)).documentElement

                    items = dom_tree.getElementsByTagName("Item")
                    bboxes, clses = [], []
                    for item in items:
                        x1 = clip(float(_get_elem_from_tree(item, "x1")))
                        y1 = clip(float(_get_elem_from_tree(item, "y1")))
                        x2 = clip(float(_get_elem_from_tree(item, "x2")))
                        y2 = clip(float(_get_elem_from_tree(item, "y2")))
                        cls = str(_get_elem_from_tree(item, "cls"))

                        bboxes.append([x1, y1, x2, y2])
                        clses.append(cls)
                    output_texts[i]["value"] = dict(bboxes=torch.tensor(bboxes), clses=clses)

        return dict(
            vision_paths=vision_paths, 
            output_texts=output_texts, 
            task_type=task_type, 
            vision_type='image')

    def get_3d_data(self, index):
        output_texts = self.data_3d['caption_list'][index]
        task_type = self.data_3d['task_type_list'][index]
        vision_embeds_3d_ref = self.data_3d['vision_embeds_3d_ref_list'][index]
        vision_embeds_3d_scene_prop = self.data_3d['vision_embeds_3d_scene_prop_list'][index]
        vision_pos_3d_ref = self.data_3d['vision_pos_3d_ref_list'][index]
        vision_pos_3d_scene_prop = self.data_3d['vision_pos_3d_scene_prop_list'][index]

        scene_id = self.data_3d['scene_id_list'][index] if len(self.data_3d['scene_id_list']) > 0 else None
        max_proposal_num = self.data_3d['max_proposal_num']
    
        vision_embeds_3d_scene_prop_padding = torch.zeros(max_proposal_num, vision_embeds_3d_scene_prop.shape[-1])
        vision_embeds_3d_scene_prop_padding[:vision_embeds_3d_scene_prop.shape[0]] = vision_embeds_3d_scene_prop
        
        vision_pos_3d_scene_prop_padding = torch.zeros(max_proposal_num, vision_pos_3d_scene_prop.shape[-1])
        vision_pos_3d_scene_prop_padding[:vision_embeds_3d_scene_prop.shape[0]] = vision_pos_3d_scene_prop
        
        mask = torch.zeros(max_proposal_num)
        mask[:vision_embeds_3d_scene_prop.shape[0]] = 1
        return dict(
            output_texts=output_texts,
            vision_type='pcl',
            task_type=task_type,
            vision_embeds_3d_ref=vision_embeds_3d_ref.reshape(-1),
            vision_embeds_3d_scene_prop=vision_embeds_3d_scene_prop_padding,
            vision_pos_3d_ref=vision_pos_3d_ref.reshape(-1),
            vision_pos_3d_scene_prop=vision_pos_3d_scene_prop_padding,
            mask=mask,
            scene_id=scene_id,
        )

    def prepare_2d_data(self):  
        vision_path_list, caption_list, task_type_list = [], [], []
        with open(self.data_file_path_2d, 'r') as fr:
            json_data = json.load(fr)
        
        vision_type = 'image'
        for item in json_data:
            if vision_type not in item:
                continue
            one_vision_name, one_caption = item[vision_type], item['conversations']
            task_type = item['task_type'] if 'task_type' in item else 'normal'

            if not one_vision_name.startswith('/'):
                one_vision_path = os.path.join(self.vision_root_path_2d, one_vision_name)
            else:
                one_vision_path = one_vision_name
            
            vision_path_list.append(one_vision_path)
            caption_list.append(one_caption)
            task_type_list.append(task_type)

        print(f'[!] collect {len(vision_path_list)} samples (loop x{self.loop_2d}) for image modality training')
        return dict(
            vision_path_list=vision_path_list, 
            caption_list=caption_list, 
            task_type_list=task_type_list)
    
    def prepare_3d_data(self):
        with open(self.data_file_path_3d, 'r') as f:
            json_data = json.load(f)

        pickle_path = os.path.join(self.vision_root_path_3d, 'scan2inst_train.pickle')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            print(f'[!] use cache from {pickle_path} for 3d data')

        else:
            vision_embeds_3d_ref_list, vision_embeds_3d_scene_prop_list = [], []
            vision_pos_3d_ref_list, vision_pos_3d_scene_prop_list = [], []
            caption_list, task_type_list = [], []
            scene_id_list = []

            max_proposal_num = 0
            scene_id_to_3d_embeds = {}
            scene_id_to_3d_pos = {}
            scene_id_to_2d_embeds = {}
            scene_id_to_scene_scale = {}

            for scene_id in tqdm(os.listdir(os.path.join(self.vision_root_path_3d, 'lamm_scannet_tr3d', 'ins_pc_feat')), desc="generate scene features"):
                # scene-level 3d prop vision embeds
                scene_prop_feat_3d_root = os.path.join(self.vision_root_path_3d, 'lamm_scannet_tr3d', 'ins_pc_feat', scene_id)
                obj_prop_path_list = sorted(os.listdir(scene_prop_feat_3d_root))
                scene_scale_root = os.path.join(self.vision_root_path_3d, 'scannet_scale', scene_id+'.npy')
                scene_id_to_scene_scale[scene_id] = torch.tensor(np.load(scene_scale_root), dtype=torch.float32)
                max_proposal_num = max(max_proposal_num, len(obj_prop_path_list))

                scene_gt_3d_feat = []
                for obj_prop_path in obj_prop_path_list:
                    scene_gt_3d_feat.append(torch.tensor(np.load(os.path.join(scene_prop_feat_3d_root, obj_prop_path)), dtype=torch.float16)) 
                scene_id_to_3d_embeds[scene_id] = torch.stack(scene_gt_3d_feat)
                
                # scene-level 3d prop pos, we need to convert (8, 3) to center+size (6,)
                scene_prop_pos_3d_root = os.path.join(self.vision_root_path_3d, 'lamm_scannet_tr3d', 'bbox', scene_id)
                scene_prop_pos_3d = []
                for obj_prop_path in obj_prop_path_list:
                    obj_prop_id, obj_prop_name = obj_prop_path.split('.')[0].split('-')
                    obj_prop_bbox = np.load(os.path.join(scene_prop_pos_3d_root, f'{obj_prop_id}-{obj_prop_name}.npy'))
                    scene_prop_pos_3d.append(torch.tensor(np.concatenate([obj_prop_bbox.min(axis=0), obj_prop_bbox.max(axis=0)]), dtype=torch.float16))
                scene_id_to_3d_pos[scene_id] = torch.stack(scene_prop_pos_3d)

                # scene-level 2d vision embeds
                scene_prop_2d_root = os.path.join(self.vision_root_path_3d, 'instance_level_image_feat', scene_id)
                scene_prop_2d_feat = []
                for obj_prop_path in obj_prop_path_list:
                    obj_prop_id, obj_prop_name = obj_prop_path.split('.')[0].split('-')
                    scene_prop_2d_sub_dir = f'{obj_prop_id}_{obj_prop_name}'
                    obj_prop_image_feat_path = [p for p in os.listdir(os.path.join(scene_prop_2d_root, scene_prop_2d_sub_dir)) if p.startswith('0_') and p.endswith('.npy')][0]
                    scene_prop_2d_feat.append(torch.tensor(np.load(os.path.join(scene_prop_2d_root, scene_prop_2d_sub_dir, obj_prop_image_feat_path)), dtype=torch.float16).squeeze(0))
                scene_id_to_2d_embeds[scene_id] = torch.stack(scene_prop_2d_feat)
                
            for item in tqdm(json_data, desc='loading 3d training data'):
                task_type, caption = item.get('task_type', 'normal'), item['conversations']
                caption_list.append(caption)
                task_type_list.append(task_type)
                scene_id = item['scene_id']
                scene_id_list.append(scene_id)
                
                ###############################
                # Deal with Scene level Input #
                ###############################
                
                vision_embeds_3d_scene_prop_list.append(scene_id_to_3d_embeds[item['scene_id']])
                vision_pos_3d_scene_prop_list.append(scene_id_to_3d_pos[item['scene_id']])
                
                #############################
                # Deal with Obj level Input #
                #############################
                
                # reference object info, vqa task has no reference object
                if task_type == 'VQA3D':
                    ref_obj_name = ref_obj_id = None
                    vision_embeds_3d_ref_list.append(torch.tensor(np.zeros(768), dtype=torch.float16))
                    vision_pos_3d_ref_list.append(torch.tensor(np.zeros(6), dtype=torch.float16))
                else:
                    ref_obj_name = item['object_name']
                    ref_obj_id = item['object_id']
                
                    # obj-level 3d prop vision embeds
                    vision_embeds_3d_ref = torch.tensor(np.load(os.path.join(self.vision_root_path_3d, 'lamm_scannet_gt', 'ins_pc_feat', scene_id, f'{ref_obj_id}-{ref_obj_name}.npy')), dtype=torch.float16)
                    vision_embeds_3d_ref_list.append(vision_embeds_3d_ref.reshape(-1))
                    
                    # obj-level 3d prop pos, we need to convert (8, 3) to center+size (6,)
                    vision_pos_3d_ref = np.load(os.path.join(self.vision_root_path_3d, 'lamm_scannet_gt', 'bbox', scene_id, f'{ref_obj_id}-{ref_obj_name}.npy'))
                    vision_pos_3d_ref = torch.tensor(np.concatenate([vision_pos_3d_ref.min(axis=0), vision_pos_3d_ref.max(axis=0)]), dtype=torch.float16)
                    vision_pos_3d_ref_list.append(vision_pos_3d_ref.reshape(-1))
            
            data = {}
            data['caption_list'] = caption_list
            data['task_type_list'] = task_type_list
            data['vision_embeds_3d_ref_list'] = vision_embeds_3d_ref_list
            data['vision_embeds_3d_scene_prop_list'] = vision_embeds_3d_scene_prop_list
            data['vision_pos_3d_ref_list'] = vision_pos_3d_ref_list
            data['vision_pos_3d_scene_prop_list'] = vision_pos_3d_scene_prop_list
            data['max_proposal_num'] = max_proposal_num
            data['scene_id_list'] = scene_id_list
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)

        print(f'[!] collect {len(data["task_type_list"])} samples (loop x{self.loop_3d}) for point cloud modality training')
        return data

    def collate(self, instances):
        """collate function for dataloader"""
        vision_types = [instance['vision_type'] for instance in instances]

        instances_2d, instances_3d = [], []
        for i, vision_type in enumerate(vision_types):
            assert vision_type in self.vision_type
            if vision_type == 'image':
                instances_2d.append(instances[i])
            else:
                instances_3d.append(instances[i])
        
        return_dict = {}
        if len(instances_2d) > 0:
            instances_2d = self.collate_2d(instances_2d)
            return_dict['image'] = instances_2d
        else:
            return_dict['image'] = None
        
        if len(instances_3d) > 0:
            instances_3d = self.collate_3d(instances_3d)
            return_dict['pcl'] = instances_3d
        else:
            return_dict['pcl'] = None
        return return_dict
    
    def collate_2d(self, instances):
        vision_paths, output_texts, task_type = tuple(
            [instance[key] for instance in instances]
            for key in ("vision_paths", "output_texts", "task_type")
        )
        return dict(
            vision_paths=vision_paths,
            output_texts=output_texts,
            vision_type='image',
            task_type=task_type,
        )

    def collate_3d(self, instances):
        keys = [key for key in instances[0].keys() if key != 'vision_type']
        return_dict = defaultdict()
        return_dict['vision_type'] = 'pcl'
        for key in keys:
            return_dict[key] = []
            for instance in instances:
                return_dict[key].append(instance[key])
            if isinstance(instance[key], torch.Tensor):
                if key == 'scene_scale':
                    return_dict[key] = torch.stack(return_dict[key])
                else:
                    return_dict[key] = torch.stack(return_dict[key]).half()
            
        return return_dict
