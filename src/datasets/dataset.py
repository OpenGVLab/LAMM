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

import copy
import os
import json
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm


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
