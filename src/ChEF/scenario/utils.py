from sentence_transformers import SentenceTransformer
from sentence_transformers import util


class Bert_Similarity:
    def __init__(self, model_path = 'sentence-transformers/all-MiniLM-L6-v2') -> None:
        self.model = SentenceTransformer(model_path).cuda()
        self.cos_func = util.pytorch_cos_sim
    
    def similarity_score(self, str1, str2):
        embedding_1 = self.model.encode(str1, convert_to_tensor=True)
        embedding_2 = self.model.encode(str2, convert_to_tensor=True)
        score = self.cos_func(embedding_1, embedding_2).item()
        return score
    
    def bert_embedding(self, str):
        return self.model.encode(str, convert_to_tensor=True)
    
    def embedding_similarity_score(self, emb1, emb2):
        score_metric = self.cos_func(emb1, emb2)
        return score_metric