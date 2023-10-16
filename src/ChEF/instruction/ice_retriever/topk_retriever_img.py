"""Topk Retriever"""
from .utils import IMG_DatasetEncoder, get_image
from .base_retriever import BaseRetriever
import torch
from torch.utils.data import DataLoader
from typing import Optional
import tqdm
import faiss
import copy
import numpy as np

from transformers import AutoFeatureExtractor, AutoModel
from typing import List, Optional


class ImageTopkRetriever(BaseRetriever):
    """Image Topk In-context Learning Retriever Class
        Class of Topk Retriever.
        
    Attributes:
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
    """
    model = None

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 seed: Optional[int] = 43,
                 ice_num: Optional[int] = 1,
                 model_ckpt: Optional[str] = 'google/vit-base-patch16-224-in21k',
                 batch_size: Optional[int] = 1,
                 **kwargs) -> None:
        super().__init__(train_dataset, test_dataset, seed, ice_num)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        img_list = self.get_corpus_from_dataset(test_dataset)
        gen_datalist = [get_image(image) for image in img_list]
        self.model_ckpt = model_ckpt
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_ckpt)
        self.model = AutoModel.from_pretrained(self.model_ckpt)

        self.encode_dataset = IMG_DatasetEncoder(gen_datalist, extractor=self.extractor)
        self.dataloader = DataLoader(self.encode_dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

        self.model = self.model.to(self.device)
        self.model.eval()

        self.index = self.create_index()

    def create_index(self):
        self.select_datalist = self.get_corpus_from_dataset(self.index_ds)
        self.select_datalist = [get_image(image) for image in self.select_datalist]
        encode_datalist = IMG_DatasetEncoder(self.select_datalist, extractor=self.extractor)
        dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.config.hidden_size))
        res_list = self.forward(dataloader, process_bar=True, information="Creating index for index set...")
        id_list = np.array([res['metadata']['id'] for res in res_list])
        self.embed_list = np.stack([res['embed'] for res in res_list])

        if hasattr(self.test_ds, 'dataset_name') and self.test_ds.dataset_name == 'MMBench': # remove ices with the same question
            remove_list = self.test_ds.circularidx
            id_list = np.array([res['metadata']['id']  for res in res_list if res['metadata']['id'] not in remove_list])
            self.embed_list = np.stack([res['embed'] for res in res_list if res['metadata']['id'] not in remove_list])
        index.add_with_ids(self.embed_list, id_list) 
        
        return index

    def knn_search(self, ice_num):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]
        for entry in tqdm.tqdm(res_list):
            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, ice_num + 1)[1][0].tolist() # n 
            near_ids = self.process_list(near_ids, idx, ice_num)
            rtr_idx_list[idx] = near_ids
        return rtr_idx_list

    def forward(self, dataloader, process_bar=False, information = ''):
        print(information)
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        if process_bar:
            _dataloader = tqdm.tqdm(_dataloader)
        for _, entry in enumerate(_dataloader):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                pixel_values = torch.stack(entry['pixel_values'], dim=0).to(self.device)
                res = self.model(pixel_values).last_hidden_state[:, 0].detach().cpu().numpy()
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def retrieve(self):
        return self.knn_search(self.ice_num)
    
    def get_corpus_from_dataset(self, dataset) -> List[List]:
        image_corpus = []
        for entry in dataset:
            image_corpus.append(entry['image_path'])
        return image_corpus