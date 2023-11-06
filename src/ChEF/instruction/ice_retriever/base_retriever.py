from typing import List, Optional



class BaseRetriever:
    """Basic In-context Learning Retriever Class
        Base class for In-context Learning Retriever, without any retrieval method.
        
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
    """
    index_ds = None
    test_ds = None

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 seed: Optional[int] = 43,
                 ice_num: Optional[int] = 1,
                 **kwargs) -> None:
        self.ice_num = ice_num
        self.index_ds = train_dataset
        self.test_ds = test_dataset
        self.seed = seed
        self.fixed_ice = None

    def retrieve(self) -> List[List]:
        """
            Retrieve for each data in generation_ds.
            
        Returns:
            `List[List]`: the index list of in-context example for each data in `test_ds`.
        """
        raise NotImplementedError("Method hasn't been implemented yet")
    
    def get_corpus_from_dataset(self, dataset) -> List[List]:
        corpus = []

        assert 'question' in dataset[0], 'No question in scenarios. You should not use topk_text retriever as the questions in instruction are the same. '
        for entry in dataset:
            corpus.append(entry['question'])

        return corpus
    
    def genetate_ice(self, ice_indices: List[List[int]], prompt: List[str], inferencer_type: str = 'default'):
        ice_batch = []
        for indices in ice_indices:
            ices = []
            for i in indices:
                ice = self.index_ds[i]
                if 'question' not in ice:
                    if inferencer_type == 'icl_ppl':
                        ice['question'] = prompt[0][0]
                    else:
                        ice['question'] = prompt[0]
                ices.append(ice)
            ice_batch.append(ices)
        return ice_batch
    
    def process_list(self, near_ids, idx, ice_num):
        if idx in near_ids:
            near_ids.remove(idx)
        else:
            near_ids = near_ids[:ice_num]
        return near_ids
