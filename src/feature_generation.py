import logging
import random

import numpy as np
from numpy.linalg import norm, matrix_rank

from model_embedding import BlackModelEmbed
from config import START_SEQ


class FeatureGeneration():

    def __init__(self,
                 core_categories_list,model=None):

        self.core_categories_list=core_categories_list
        self.sbert=BlackModelEmbed(model)
        self.sbert_core_embedding_matrix=self.createEmbeddingToBertBaseUsingCore()




    def createEmbeddingToBertBaseUsingCore(self):
        basisInExpEmbedding=[]
        for c in self.core_categories_list:
            c_embedd = self.sbert.get_bert_embed(c.replace("Category:", START_SEQ).replace("_", " "))
            basisInExpEmbedding.append(c_embedd)
        basisInExpEmbedding=np.array(basisInExpEmbedding)
        return basisInExpEmbedding




    def get_features_Sbert(self,sentence:str)->np.ndarray:
        embedding=self.sbert.get_bert_embed(sentence)
        return embedding

    def get_features_explainable_sbert(self,sentence:str)->np.ndarray:
        embedding=self.sbert.get_bert_embed(sentence)
        return self.sbert_core_embedding_matrix.dot(embedding)

