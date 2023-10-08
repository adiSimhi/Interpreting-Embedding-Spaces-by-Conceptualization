from typing import List

from core_on_demand import coreOnDemand

from feature_generation import FeatureGeneration
from config import START_SEQ


class ExamplesOfModel():

    def __init__(self,core_all_depth, graph_g_path,text_path):
        self.all_core_depth=core_all_depth
        self.core_on_demand=coreOnDemand(core_all_depth, graph_g_path)
        for path in text_path:
            with open(path) as f:
                self.text = f.read()
            self.text=self.text.split(".")
            self.text=[i for i in self.text if len(i)>1]
            self.text=self.text[:10]+[self.text[-1]]
            # print(self.text)
            self.get_top_text()

    def get_category_depth(self,category):
        category="Category:"+category.replace(START_SEQ,"")
        for i,core in enumerate(self.all_core_depth):
            if category in core:
                return i+1

    def get_top_text(self):
        core=self.core_on_demand.create_core_on_demand(self.text[:-1],max_size=768)

        core= [c.replace("Category:", START_SEQ).replace("_", " ") for c in core if
                               'Wikipedia categories named after' not in c]
        self.feature_generation = FeatureGeneration(core)
        top_features = self.get_top(self.feature_generation.get_features_explainable_sbert(self.text[-1]))
        common_categories = [core[index] for index in
                             top_features]
        print(f"top features: {common_categories}")
        category_index=[]
        for cat in common_categories:
            category_index.append(self.get_category_depth(cat))
        print(f"{category_index=}")
        print(f"sentence: {self.text[-1]}")


    def get_top(self, word_embedding, number=5)->List[int]:
        """
        create the top argmax of an embedding
        :param word_embedding:
        :param number: number of argmax
        :return: list of the highest argmax
        """
        highest = word_embedding.argsort()[-number:][::-1]
        return highest


