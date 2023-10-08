import logging

from typing import List, Tuple

import networkx as nx
from scipy.spatial import distance
from scipy.stats import rankdata
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

from model_embedding import BlackModelEmbed
import numpy as np

from config import START_SEQ
from using_pickle import from_yaml_to_python, read_zip
from sklearn import tree




class coreOnDemand():

    def __init__(self, core_all_depth: List[List[str]], graph_g_path):
        self.all_cores = core_all_depth
        self.sbert = BlackModelEmbed()
        self.sbert_core_embedding = {}
        self.sbert_core_embedding_with_context = {}
        self.g_graph = read_zip(graph_g_path)
        self.percentage_of_children_to_open = 0.5
        for core in self.all_cores:
            for c in core:
                self.sbert_core_embedding[c] = self.sbert.get_bert_embed(c.replace("Category:", START_SEQ).replace("_", " "))
                self.sbert_core_embedding_with_context[c]=self.sbert.get_bert_embed(self.concept_name_with_its_children([c])[0])
        self.sbert_core_embedding_keys = self.sbert_core_embedding.keys()




    def get_features_explainable_sbert(self, matrix, sentence: str) -> np.ndarray:
        """
        get interpretable embedding given the matrix to use to move the the conceptual world.
        :param matrix:
        :param sentence:
        :return:
        """
        embedding = self.sbert.get_bert_embed(sentence)
        return matrix.dot(embedding)

    def concept_name_with_its_children(self,list_of_concepts:List[str])->List[str]:
        new_concept_names=[]
        for c in list_of_concepts:
            children=self.choose_children_to_open(c, 1)
            children=children[:min(len(children),2)]
            if len(children)<2:
                new_concept_names.append(c)
            else:
                new_concept_names.append(c.replace("Category:", START_SEQ).replace("_", " ")+" such as "+children[0].replace("Category:", START_SEQ).replace("_", " ")+" and "+children[1].replace("Category:", START_SEQ).replace("_", " "))
        # print(new_concept_names)
        return new_concept_names

    def create_core_on_demand(self, X_train, number_of_classes=None, y_train=None, w=0.8, N=1000,min_dist=None,max_size=None,remove_p=False) -> List[str]:
        """
        create core on demand that stops when N iteration is reached.
        :param X_train: list of sentences
        :param number_of_classes:in classification can add number of classification
        :param y_train: in classification can add y labels.
        :param w:in classification to use entropy and max score with w
        :param N: number of iterations
        :param min_dist:stop iterations when the distance between two most similar sentences is more then that
        :return: new core
        """
        core_opend = []
        core_names = self.all_cores[0][:]
        core_embedding = np.array([self.sbert_core_embedding.get(c) for c in core_names])
        core_on_train = [self.get_features_explainable_sbert(core_embedding, x) for x in X_train]
        if number_of_classes is not None:
            entropy_core = self.calculate_entropy(core_names, y_train, core_on_train, number_of_classes)
        assert len(core_names) == len(core_embedding)
        for i in range(N):
            if max_size is not None and len(core_names)>=max_size:
                break
            average_train_vector = np.average(np.array(core_on_train), axis=0)
            if number_of_classes is not None:
                entropy_mul_score = entropy_core * (1 - w) + w * average_train_vector
            else:
                entropy_mul_score = average_train_vector
            core_to_expand = self.find_core_to_expend(core_opend, entropy_mul_score, core_names)
            if core_to_expand == "":
                break
            core_opend.append(core_to_expand)
            core_children = self.choose_children_to_open(core_to_expand, self.percentage_of_children_to_open)
            core_children = [c for c in core_children if c not in core_names and c in self.sbert_core_embedding_keys]

            core_names += core_children
            #remove parent
            if remove_p is True and len(core_children)>0:
                core_names.remove(core_to_expand)
            core_embedding = np.array([self.sbert_core_embedding.get(c) for c in core_names])
            assert len(core_names) == len(core_embedding)
            core_on_train = [self.get_features_explainable_sbert(core_embedding, x) for x in X_train]
            # print(f"{len(core_names)=}, {len(core_embedding)=},{np.shape(core_on_train)=}\n{core_to_expand=},{core_children=}")
            # the information gain is not improving thus we can stop
            if number_of_classes is not None and (len(core_children)>0):
                new_entropy_core = self.calculate_entropy(core_names, y_train, core_on_train, number_of_classes)
                if np.average(new_entropy_core)<=1/number_of_classes and max_size is None:
                    break
                entropy_core=new_entropy_core
            # large distance even in the two most similar sentences, thus we can stop openning categories
            if min_dist is not None and (len(core_children)>0):
                distence_min=sorted(set(distance.cdist(np.array(core_on_train), np.array(core_on_train), 'cosine').flatten()))[0]
                if distence_min>min_dist:
                    break

        print(f"final core length {len(core_names)}")
        return core_names


    def calculate_entropy(self,core_names,y_train,core_on_train,number_of_classes):
        """
        for each category in core check it's entropy. to each train sample it calculate the highest core category and adds the
        y of the train sample to the y's of this core category. We hope that most core categories will have
        not a lot of different y's that are correlated to them
        :param core_names:
        :param y_train:
        :param core_on_train:
        :param number_of_classes:
        :return: entropy of the core categories
        """
        classes_per_core = [[] for i in range(len(core_names))]
        for i, y in enumerate(y_train):
            core_max = core_on_train[i].argmax()
            classes_per_core[core_max].append(y)
        entropy_core = np.array([len(set(classes)) / number_of_classes for classes in classes_per_core])
        return entropy_core


    def find_core_to_expend(self, list_of_core_opened: List[str], score_core: np.ndarray, core_list: List[str]) -> str:
        """
        find the next category to open that was not open yet based on the categories score.
        :param list_of_core_opened:opened categories
        :param score_core: score of the categories
        :param core_list: core categories list
        :return:category to open next
        """
        score_core_arg = score_core.argsort()[::-1]
        best_core = ""
        for score in score_core_arg:
            if core_list[score] not in list_of_core_opened:
                best_core = core_list[score]
                break
        return best_core

    def choose_children_to_open(self, node_name, number_to_choose):
        """
        choosing the children of the category to open based on the score of the edge between the child and the category
        :param node_name: category to open
        :param number_to_choose: percentage of children to open
        :return: list of children of the category to add to core.
        """
        final_children = []
        node_edges = self.g_graph.out_edges(node_name, data=True)
        for node in node_edges:
            if "Category:" in node[1]:
                weight = node[2].get("weight", 1)
                final_children.append((node[1], weight))
        final_children = sorted(final_children, key=lambda x: x[1])
        final_children = final_children[-round(number_to_choose * len(final_children)):]
        return [self.remove_non_ascii_chars(c[0]) for c in final_children]

    def remove_non_ascii_chars(self,string):
        return ''.join([i if ord(i) < 128 else ' ' for i in string])