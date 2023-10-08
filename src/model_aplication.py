from typing import List

from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from feature_generation import FeatureGeneration
import numpy as np

from config import START_SEQ


class checkPhrases():

    def __init__(self,core:List[str]):

        self.core =  self.core = [c.replace("Category:", START_SEQ).replace("_", " ") for c in core if
                'Wikipedia categories named after' not in c]
        print(f"{len(core)=}")
        self.st5model = SentenceTransformer("sentence-t5-large")
        self.sbertmodel = SentenceTransformer("bert-base-nli-mean-tokens")
        self.distilmodel = SentenceTransformer("all-distilroberta-v1")
        self.model_list = {"bert-base-nli-mean-tokens": self.sbertmodel,
                           "all-distilroberta-v1": self.distilmodel,"sentence-t5-large": self.st5model}

    def get_top(self, word_embedding, number=5)->List[int]:
        """
        create the top argmax of an embedding
        :param word_embedding:
        :param number: number of argmax
        :return: list of the highest argmax
        """
        highest = word_embedding.argsort()[-number:][::-1]
        return highest

    def get_top_text(self,phrase:str,model):

        self.feature_generation = FeatureGeneration(self.core,model)
        top_features = self.get_top(self.feature_generation.get_features_explainable_sbert(phrase))
        common_categories = [self.core[index] for index in
                             top_features]
        print(f"{phrase=}, top features= {common_categories}")

    def get_phrase_top(self,phrases:List[str]):
        for p in phrases:
            for name, model in self.model_list.items():
                print(f"model name={name}")
                self.get_top_text(p,name)

    def cosine_similarity(self, embedding_vector1, embedding_vector2):
        return np.dot(embedding_vector1, embedding_vector2) / (norm(embedding_vector1) * norm(embedding_vector2))

    def triple_check(self,triple_phrases):

        for name, model in self.model_list.items():
            model_triple_results = []
            for t in triple_phrases:
                sim1 = self.cosine_similarity(model.encode(t[0], show_progress_bar=False, normalize_embeddings=True),
                                              model.encode(t[1], show_progress_bar=False, normalize_embeddings=True))
                sim2 = self.cosine_similarity(model.encode(t[0], show_progress_bar=False, normalize_embeddings=True),
                                              model.encode(t[2], show_progress_bar=False, normalize_embeddings=True))
                model_triple_results.append((t, sim1, sim2))
            print(f"model {name} similarity in the following triples:{model_triple_results}")
