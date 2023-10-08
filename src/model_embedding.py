from sentence_transformers import SentenceTransformer
from config import MODEL_NAME
import random
import numpy as np
class BlackModelEmbed():

    def __init__(self,model=None):
        random.seed(10)
        self.word_dist_embedding={}
        if model is None:
            self.model = SentenceTransformer(MODEL_NAME)
        else:
            self.model = SentenceTransformer(model)


    def get_bert_embed(self,word:str):
        embed=self.word_dist_embedding.get(word)
        if embed is not None:
            return embed
        embed=self.create_embedding(word)
        self.word_dist_embedding[word]=embed
        return embed


    def create_embedding(self,word:str):
        sentence_embeddings = self.model.encode(word,show_progress_bar=False,normalize_embeddings=True)
        return sentence_embeddings

