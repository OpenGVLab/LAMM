from .random_retriever import RandomRetriever
from .topk_retriever import TopkRetriever
from .fixed_retriever import FixedRetriever
from .topk_retriever_img import ImageTopkRetriever


retriever_dict = {
    'random': RandomRetriever,
    'topk_text': TopkRetriever,
    'fixed': FixedRetriever,
    'topk_img': ImageTopkRetriever
}

def build_retriever(train_dataset, test_dataset, retriever_type, **kwargs):
    build_fuc = retriever_dict[retriever_type]
    return build_fuc(train_dataset, test_dataset, **kwargs)