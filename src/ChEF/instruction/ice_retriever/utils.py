from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import AutoTokenizer
from transformers import AutoFeatureExtractor
import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.file_utils import PaddingStrategy
import numpy as np
from PIL import Image


def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    
    elif isinstance(image, Image.Image):
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image)
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


class ListWrapper:
    def __init__(self, data: List[Any]):
        self.data = data

    def to(self, device):
        return self.data


def ignore_pad_dict(features):
    res_dict = {}
    if "metadata" in features[0]:
        res_dict['metadata'] = ListWrapper([x.pop("metadata") for x in features])
    return res_dict


@dataclass
class DataCollatorWithPaddingAndCuda:
    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchEncoding:
        res_dict = ignore_pad_dict(features)

        has_labels = "labels" in features[0]
        if has_labels:
            labels = [{"input_ids": x.pop("labels")} for x in features]
            labels = self.tokenizer.pad(
                labels,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
                verbose=False
            )

        # print(features)
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
            verbose=False
        )

        if has_labels:
            batch['labels'] = labels.input_ids
        batch.update(res_dict)

        if self.device:
            batch = batch.to(self.device)

        return batch

class DatasetEncoder(torch.utils.data.Dataset):
    def __init__(self, datalist: List, model_name=None, tokenizer=None) -> None:
        self.datalist = datalist
        if model_name is None and tokenizer is None:
            raise ValueError("model_name and tokenizer could not both be None")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
        self.encode_dataset = []
        self.init_dataset()
        self.datalist_length = len(self.encode_dataset)

    def init_dataset(self):
        for idx, data in enumerate(self.datalist):
            tokenized_data = self.tokenizer.encode_plus(data, truncation=True, return_tensors='pt', verbose=False)
            self.encode_dataset.append({
                'input_ids': tokenized_data.input_ids[0],
                'attention_mask': tokenized_data.attention_mask[0],
                "metadata": {"id": idx, "len": len(tokenized_data.input_ids[0]),
                             "text": data}
            })

    def __len__(self):
        return self.datalist_length

    def __getitem__(self, idx):
        return self.encode_dataset[idx]


class IMG_DatasetEncoder(torch.utils.data.Dataset):
    def __init__(self, datalist: List, model_name=None, extractor=None) -> None:
        self.datalist = datalist
        if model_name is None and extractor is None:
            raise ValueError("model_name and extractor could not both be None")
        if extractor is not None:
            self.extractor = extractor
        else:
            self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.encode_dataset = []
        self.init_dataset()
        self.datalist_length = len(self.encode_dataset)

    def init_dataset(self):
        for idx, data in enumerate(self.datalist):
            img_feature = self.extractor(data, return_tensors='pt')
            self.encode_dataset.append({
                'pixel_values': img_feature.pixel_values[0],
                "metadata": {"id": idx, "img": data}
            })

    def __len__(self):
        return self.datalist_length

    def __getitem__(self, idx):
        return self.encode_dataset[idx]