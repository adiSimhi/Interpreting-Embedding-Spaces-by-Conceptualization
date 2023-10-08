import csv
import random
from typing import List, Tuple
import numpy as np
from numpy.linalg import norm

from feature_generation import FeatureGeneration


class TripletsTest():

    def __init__(self, classification_path: str,
                 core_categories_list:List[List[str]]):
        self.article_section_articles={}
        self.core_categories=core_categories_list[2]
        print(f"number of categories {len(self.core_categories)}")
        self.classification_path=classification_path
        self.all_x_all_y = self.get_data()
        self.all_x = [x_y[0] for x_y in self.all_x_all_y]
        self.all_y = [x_y[1] for x_y in self.all_x_all_y]
        self.all_depth_info=[]
        self.feature_sbert=FeatureGeneration(self.core_categories)
        self.remove_sections()


    def get_data(self) -> List[Tuple]:
        """
        getting all the data from the file
        :return:
        """
        x_y = []
        xis=[]
        self.articles_names=[]
        with open(self.classification_path, newline='') as f:
            data = csv.reader(f, skipinitialspace=True)
            for i, row in enumerate(data):
                tuple_value = (row[1],row[0], row[2])
                if tuple_value not in x_y and row[1] not in xis:
                    x_y.append(tuple_value)
                    xis.append(row[1])
                    self.articles_names.append(row[0])
                    self.add_sentence_to_article_section_dict(row[1],row[0],row[2])
        self.articles_names=list(set(self.articles_names))
        return x_y


    def add_sentence_to_article_section_dict(self,sentence,article,section):
        article_=self.article_section_articles.get(article)
        if article_ is None:
            self.article_section_articles[article]={}
        section_=self.article_section_articles[article].get(section)
        if section_ is None:
            self.article_section_articles[article][section]=[]
        self.article_section_articles[article][section]+=[sentence]


    def remove_sections(self):
        articles_to_remove=[]
        for article,sections in self.article_section_articles.items():
            if len(sections)<=1:
                articles_to_remove.append(article)
        for article in articles_to_remove:
            self.article_section_articles.pop(article)

    def random_seed(self,i):
        #12 for c3 18 for c2
        random.seed(i)

    def generate_triples_similar_categories(self) -> List[Tuple]:
        """
        generate triple of examples
        :return:examples trio
        """

        samples_trio = []
        for i in range(1400):
            self.random_seed(i)
            samples=random.sample(list(self.article_section_articles.keys()), 1)
            self.random_seed(i)
            sample_two_sections=random.sample(list(self.article_section_articles.get(samples[0]).keys()),2)
            self.random_seed(i)
            samples_from_same_category=random.sample(self.article_section_articles.get(samples[0]).get(sample_two_sections[0]),2)
            self.random_seed(i)
            sample_from_different_category=random.sample(self.article_section_articles.get(samples[0]).get(sample_two_sections[1]),1)
            if (samples_from_same_category[0],samples_from_same_category[1],sample_from_different_category[0]) not in samples_trio\
                   and (samples_from_same_category[1],samples_from_same_category[0],sample_from_different_category[0]) not in samples_trio:
                samples_trio.append((samples_from_same_category[0],samples_from_same_category[1],sample_from_different_category[0]))

        return samples_trio

    def cosine_similarity(self, embedding_vector1, embedding_vector2):
        return np.dot(embedding_vector1, embedding_vector2) / (norm(embedding_vector1) * norm(embedding_vector2))


    def get_similarity_classification(self,sample_triple,using_sbert_explanation_func)->List:
        classification=[]
        for sample in sample_triple:
            sent0_embed=using_sbert_explanation_func(sample[0])
            sent1_embed=using_sbert_explanation_func(sample[1])
            sent2_embed=using_sbert_explanation_func(sample[2])

            sentence0_1_sim=self.cosine_similarity(sent0_embed,sent1_embed)
            sentence0_2_sim=self.cosine_similarity(sent0_embed,sent2_embed)
            if sentence0_2_sim>sentence0_1_sim:
                classification.append(1)
            else:
                classification.append(0)
        return classification

    def kapper_test(self,y1,y2):
        p0=sum([1 for j in range(len(y1)) if y1[j]==y2[j]])/len(y1)
        classes=list(set(y1).union(y2))
        pe=0
        for k in classes:
            sm=sum([1 for j in range(len(y1)) if y1[j]==k])*sum([1 for j in range(len(y2)) if y2[j]==k])
            pe+=sm
        pe=pe/len(y1)**2
        k=(p0-pe)/(1-pe)

        return p0,k,pe


    def random_swith_model_results(self,sample_from_examples_randomly,list):
        new_list=[]
        for i,val in enumerate(list):
            if i in sample_from_examples_randomly:
                new_list.append((val+1)%2)
            else:
                new_list.append(val)
        return new_list

    def agreement_on_triplets(self):

        sample_triples=self.generate_triples_similar_categories()
        sample_triples=sample_triples[:min(1000,len(sample_triples))]
        print(f"number of samples {len(sample_triples)}")
        real_similarity=[0]*len(sample_triples)
        sbert_similarity=self.get_similarity_classification(sample_triples,self.feature_sbert.get_features_Sbert)
        sbert_exp_similarity=self.get_similarity_classification(sample_triples,self.feature_sbert.get_features_explainable_sbert)
        self.random_seed(1)
        sample_from_examples_randomly=random.sample([i for i in range(len(real_similarity))],int(len(real_similarity)/2))
        real_similarity=self.random_swith_model_results(sample_from_examples_randomly,real_similarity)

        sbert_similarity=self.random_swith_model_results(sample_from_examples_randomly,sbert_similarity)
        sbert_exp_similarity=self.random_swith_model_results(sample_from_examples_randomly,sbert_exp_similarity)
        print(f"agreement and kappa between CES and sbert is {self.kapper_test(sbert_similarity, sbert_exp_similarity)}")
        print(f"agreement and kappa between CES and True is {self.kapper_test(sbert_exp_similarity, real_similarity)}")
        print(f"agreement and kappa between sbert and True is {self.kapper_test(sbert_similarity, real_similarity)}")


