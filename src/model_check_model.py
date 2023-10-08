import logging
from typing import List, Tuple

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
class ModelCheckModel():
    def __init__(self):
        self.model_sbert = SentenceTransformer("bert-base-nli-mean-tokens")
        self.model_st5 = SentenceTransformer("sentence-t5-large")
        self.model_distil = SentenceTransformer("all-distilroberta-v1")

    def get_check_model_embedding(self,sentence,model):
        return model.encode(sentence, show_progress_bar=False, normalize_embeddings=True)

    def cosine_similarity(self, embedding_vector1, embedding_vector2)->float:
        """
        calculate cosine similarity of two embedding vectors
        :param embedding_vector1:
        :param embedding_vector2:
        :return: cosin similarity
        """
        return np.dot(embedding_vector1, embedding_vector2) / (norm(embedding_vector1) * norm(embedding_vector2))

    def check_top_similarity_to_sentence(self, top: List[str], sentence: str,model) -> float:
        """
        similarity check by another model
        :param top:
        :param sentence:
        :return:
        """
        average_similarity = []
        sentence_embedd = self.get_check_model_embedding(sentence,model)
        for word in top:
            word_embedd = self.get_check_model_embedding(str(word),model)
            average_similarity.append(self.cosine_similarity(word_embedd, sentence_embedd))
        return np.average(average_similarity)


    def get_score_model_directly(self,classes,tops,model):
        accuracy_model_model=[]
        for i in range(len(tops)):
            result1 = self.check_top_similarity_to_sentence(tops[i], classes[0],model)
            result2 = self.check_top_similarity_to_sentence(tops[i], classes[1],model)
            if result2>result1:
                accuracy_model_model.append(classes[1])
            else:
                accuracy_model_model.append(classes[0])
        return accuracy_model_model


