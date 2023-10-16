from typing import Optional
from tqdm import trange
import numpy as np
from .base_retriever import BaseRetriever


class RandomRetriever(BaseRetriever):
    """Random In-context Learning Retriever Class
        Class of Random Retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        seed (`int`, optional): Seed for the random number generator.
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 ice_num: Optional[int] = 1,
                 seed: Optional[int] = 43,
                 **kwargs) -> None:
        super().__init__(train_dataset, test_dataset, seed, ice_num)

    def retrieve(self):
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        rtr_idx_list = []
        for _ in trange(len(self.test_ds)):
            idx_list = np.random.choice(num_idx, self.ice_num, replace=False).tolist()
            rtr_idx_list.append(idx_list)
        return rtr_idx_list