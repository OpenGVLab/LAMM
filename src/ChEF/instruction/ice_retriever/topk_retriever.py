"""Topk Retriever"""
from .utils import DatasetEncoder, DataCollatorWithPaddingAndCuda
from .base_retriever import BaseRetriever
import torch
from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import tqdm
import faiss
import copy
import numpy as np


class TopkRetriever(BaseRetriever):
    """Topk In-context Learning Retriever Class
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
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 seed: Optional[int] = 43,
                 ice_num: Optional[int] = 1,
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 **kwargs) -> None:
        super().__init__(train_dataset, test_dataset, seed, ice_num)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        gen_datalist = self.get_corpus_from_dataset(test_dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        self.encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        self.dataloader = DataLoader(self.encode_dataset, batch_size=self.batch_size, collate_fn=co)

        self.model = SentenceTransformer(sentence_transformers_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.index = self.create_index()

    def create_index(self):
        self.select_datalist = self.get_corpus_from_dataset(self.index_ds)
        encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))
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
            near_ids = self.index.search(embed, ice_num + 1)[1][0].tolist()
            near_ids = self.process_list(near_ids, idx, ice_num)
            rtr_idx_list[idx] = near_ids
        return rtr_idx_list

    def forward(self, dataloader, process_bar=False, information=''):
        print(information)
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        if process_bar:
            _dataloader = tqdm.tqdm(_dataloader)
        for _, entry in enumerate(_dataloader):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                raw_text = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
                res = self.model.encode(raw_text, show_progress_bar=False)
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def retrieve(self):
        return self.knn_search(self.ice_num)