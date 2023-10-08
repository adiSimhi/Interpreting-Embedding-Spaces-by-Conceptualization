import csv
import logging
import random
import string
from typing import List, Tuple

import nltk
from datasets import load_dataset
from nltk.corpus import stopwords
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from feature_generation import FeatureGeneration
from core_on_demand import coreOnDemand
import numpy as np

from config import START_SEQ
from using_pickle import from_yaml_to_python

from model_check_model import ModelCheckModel


class ModelTest():

    def __init__(self, classification_path: List[str], core_all_depth: List[List[str]], graph_g_path):
        self.coreOnDemand = coreOnDemand( core_all_depth, graph_g_path)
        self.classification_path = classification_path
        self.g=graph_g_path
        self.random_forest = RandomForestClassifier(max_depth=5, random_state=42)
        self.sample_per_class=2
        self.model_check_model = ModelCheckModel()
        self.y_real_name_ohsumed = {"C23": "Pathological Conditions, Signs and Symptoms", "C22": "Animal Diseases",
                                    "C21": "Disorders of Environmental Origin", "C20": "Immunologic Diseases",
                                    "C19": "Endocrine Diseases",
                                    "C18": "Nutritional and Metabolic Diseases",
                                    "C17": "Skin and Connective Tissue Diseases",
                                    "C16": "Neonatal Diseases and Abnormalities", "C15": "Hemic and Lymphatic Diseases",
                                    "C14": "Cardiovascular Diseases",
                                    "C13": "Female Genital Diseases and Pregnancy Complications",
                                    "C12": "Urologic and Male Genital Diseases", "C11": "Eye Diseases",
                                    "C10": "Nervous System Diseases", "C09": "Otorhinolaryngologic Diseases",
                                    "C08": "Respiratory Tract Diseases", "C07": "Stomatognathic Diseases",
                                    "C06": "Digestive System Diseases", "C05": "Musculoskeletal Diseases",
                                    "C04": "Neoplasms", "C03": "Parasitic Diseases", "C02": "Virus Diseases",
                                    "C01": "Bacterial Infections and Mycoses"}

    def random_seed(self):
        """
        create seed in random
        :return:
        """
        random.seed(42)

    def entropy_using_log(self,score):
        if score == 1 or score == 0:
            first_ig = 0
        else:
            first_ig = -score * np.log2(score) - (1 - score) * np.log2(1 - score)
        return first_ig

    def entropy(self,score):
        score=max(score,1-score)
        return score

    def information_gain_of_feature(self,x_f,yis):
        classes = list(set(yis))
        x_y = [(x_f[i], yis[i]) for i in range(len(x_f))]
        x_y_sorted = sorted(x_y, key=lambda x: x[0])
        best_ig = -1
        for i in range(1, len(x_y_sorted)):
            y_first_group = [c[1] for c in x_y_sorted[:i]]
            y_second_group = [c[1] for c in x_y_sorted[i:]]
            first_ig = (sum(1 for y in y_first_group if y == classes[0])) / len(y_first_group)
            first_ig=self.entropy(first_ig)
            second_ig = (sum(1 for y in y_second_group if y == classes[0])) / len(y_second_group)
            second_ig=self.entropy(second_ig)
            score = (first_ig * len(y_first_group) + second_ig * len(y_second_group)) / len(x_y_sorted)
            if score > best_ig:
                best_ig = score
        return best_ig

    def information_gain_all_features(self,points,core):
        cis=[]
        embedd=[]
        y=[]
        for x, embedding_y in points.items():
            x_embedding, x_y = embedding_y
            embedd.append(x_embedding)
            y.append(x_y)
        for i,c in enumerate(core):
            x_feature=[x[i] for x in embedd]
            best_ig_c=self.information_gain_of_feature(x_feature,y)
            cis.append((i,best_ig_c))
        # print(f"{cis=}")
        cis=sorted(cis,key=lambda x: x[1])
        low_cis=[c[0] for c in cis][:int(self.ig_p*len(core))]
        # print(f"{low_cis=}",flush=True)

        return low_cis





    def get_data(self, index) -> List[Tuple]:
        """
        getting all the data from the file
        :return:
        """
        x_y = []
        sentence_length = []
        xis = []
        with open(self.classification_path[index], newline='') as f:
            data = csv.reader(f, skipinitialspace=True)
            for i, row in enumerate(data):
                tuple_value = (row[0], row[1])
                if tuple_value not in x_y and row[0] not in xis:
                    x_y.append(tuple_value)
                    xis.append(row[0])
                    sentence_length.append(len(tuple_value[0].split(" ")))
        y_class = set([tuple[1] for tuple in x_y])
        my_dict = {i: [tuple[1] for tuple in x_y].count(i) for i in [tuple[1] for tuple in x_y]}
        return x_y

    def ohsumed_data(self,index)->Tuple:
        """
        get ohsumed dataset of three classes
        :param index: index in list of path that is the ohsumed path
        :return: all x values and all y values
        """
        data_processed = self.get_data(index)
        data=[(i[0],self.y_real_name_ohsumed.get(i[1])) for i in data_processed ]
        self.random_seed()
        all_x_all_y = random.sample(data, min(len(data), 10000))
        all_x = [x_y[0] for x_y in all_x_all_y]
        all_y = [x_y[1] for x_y in all_x_all_y]
        return all_x, all_y

    def get_data_else(self, index)->tuple:
        """
        get the yahoo dataset and can also get more, if their are more then 10000 samples randomly choose 10000
        :param index: index in list of path that is the wanted one
        :return: all x values and all y values
        """
        data_processed = self.get_data(index)
        self.random_seed()
        all_x_all_y = random.sample(data_processed, min(len(data_processed), 10000))
        all_x = [x_y[0] for x_y in all_x_all_y]
        all_y = [x_y[1] for x_y in all_x_all_y]
        return all_x, all_y





    def get_ag_news_data(self):
        """
        get ag news, choose randomly 10000 samples to work with
        :return: all x values and all y values
        """
        set_options={0:"world",1:"sports",2:"business",3:"science and technology"}
        train = load_dataset("ag_news")["train"]
        X_train = [word.get("text") for word in train]
        y_train = [set_options.get(word.get('label')) for word in train ]
        c = list(zip(X_train, y_train))
        self.random_seed()
        random.shuffle(c)
        X_train, y_train = zip(*c)
        return X_train[:10000],y_train[:10000]


    def create_all_points_embedding(self, x_train, y_train,func)->dict:
        """
        creating embedding to all points
        :return:
        """
        points_embedding={}
        for i, x in enumerate(x_train):
            x_embedding = func(x)
            points_embedding[x] = (x_embedding, y_train[i])
        return points_embedding

    def cosine_similarity(self, embedding_vector1, embedding_vector2)->float:
        """
        calculate cosine similarity of two embedding vectors
        :param embedding_vector1:
        :param embedding_vector2:
        :return: cosin similarity
        """
        return np.dot(embedding_vector1, embedding_vector2) / (norm(embedding_vector1) * norm(embedding_vector2))



    def get_top(self, word_embedding, number=0.15)->List[int]:
        """
        create the top argmax of an embedding
        :param word_embedding:
        :param number: number of argmax
        :return: list of the highest argmax
        """
        highest = word_embedding.argsort()[-round(number*len(word_embedding)):][::-1]
        return highest



    def get_classification_embedding(self,embedding_sentence):
        """
        get classification prediction of a point usinf it's embedding
        :param embedding_sentence:
        :return: prediction
        """
        return self.random_forest.predict([embedding_sentence])[0]



    def get_prediction(self,embedding_sentence,classes):
        """
        get classification prediction of a point using it's embedding
        :param embedding_sentence:
        :return: prediction
        """
        y_average_embedding_set=[[],[]]
        for x, embedding_y in self.point_embedding_model.items():
            x_embedding, x_y = embedding_y
            if x_y==classes[0]:
                y_average_embedding_set[0].append(x_embedding)
            else:
                y_average_embedding_set[1].append(x_embedding)
        y_average_embedding_set=[np.average(e,axis=0) for e in y_average_embedding_set]
        if self.cosine_similarity(embedding_sentence,y_average_embedding_set[0])>\
            self.cosine_similarity(embedding_sentence, y_average_embedding_set[1]):
            return classes[0]
        else:
            return classes[1]




    def get_explanation_sentence(self, sentence, classes,func_for_explanation,core,igc_low)->Tuple:
        """
        Try to find the top features that are relevant for the class that the model choose and of the sentence, if the model
        and the real class are the same returns Nones
        :param sentence: the x
        :param y: the y
        :param freq_class:classes embedding dict
        :param func_for_explanation: function for creation of icos embedding
        :param core: core categories
        :return: if y and the model class is the same tuple of Nones else, y of the model, similare top feature core,
        similarity using a different model
        """
        y_bert=self.get_classification_embedding(self.feature_generation.get_features_Sbert(sentence))
        # top_sentence = self.get_top(func_for_explanation(sentence))
        top_sentence=func_for_explanation(sentence).argsort()[::-1]
        sim_ordered=[core[val] for val in top_sentence if val not in igc_low]
        sim_ordered=sim_ordered[:min(len(sim_ordered),3)]
        return y_bert,sim_ordered




    def train_model(self,x_train,y_train):
        """
        train random forest using the model embedding
        :param x_train:
        :param y_train:
        :return:
        """
        x_features = []
        for x in x_train:
            model = self.feature_generation.get_features_Sbert(x)
            x_features.append(model)
        self.random_forest.fit(x_features, y_train)


    def create_binary_dataset(self,all_x,all_y)->Tuple:
        classes = []
        for y in all_y:
            if y not in classes:
                classes.append(y)
        self.random_seed()
        binary_classes=random.sample(classes, 2)
        assert len(binary_classes)==2
        min_sample_for_each_tag=min(sum([1 for i in all_y if i==binary_classes[0]]),
                                    sum([1 for i in all_y if i==binary_classes[1]]))
        new_x=[]
        new_y=[]
        samples_tag=[0,0]
        for i,y in enumerate(all_y):
            if y in binary_classes and samples_tag[binary_classes.index(y)]<min_sample_for_each_tag:
                new_y.append(y)
                new_x.append(all_x[i])
                samples_tag[binary_classes.index(y)]+=1
        print(f"number of tag {classes[0]} is {sum([1 for i in new_y if i==binary_classes[0]])}"
              f"and number of tags {classes[1]} is {sum([1 for i in new_y if i==binary_classes[1]])}")
        assert len(new_x)==len(new_y)

        return new_x,new_y

    def run_all_options(self,path_static,path_static_core):
        ig_p=[0.8]
        for p in ig_p:
            self.ig_p=p
            print(f"ig={self.ig_p}")
            self.run_all_tests(path_static,path_static_core)



    def run_all_tests(self,path_static,path_static_core):

        for i in range(1,len(self.classification_path)-1):
            print(f"{self.classification_path[i]} dataset",flush=True)
            all_x, all_y = self.get_data_else(i)
            all_x,all_y=self.create_binary_dataset(all_x,all_y)
            self.run_model_test_creation(all_x,all_y,path_static,path_static_core)
        all_x,all_y=self.get_ag_news_data()
        all_x, all_y = self.create_binary_dataset(all_x, all_y)
        print(f"ag_news dataset",flush=True)
        self.run_model_test_creation(all_x,all_y,path_static,path_static_core)
        all_x, all_y = self.ohsumed_data(0)
        all_x, all_y = self.create_binary_dataset(all_x, all_y)
        print(f"Ohsumed dataset",flush=True)

        self.run_model_test_creation(all_x, all_y, path_static, path_static_core)

    def kapper_test(self, y1, y2):
        p0 = sum([1 for j in range(len(y1)) if y1[j] == y2[j]]) / len(y1)
        return p0

    def get_prediction_by_lalel_concept_similarity(self,sentence_embedding, classes):
        label1=self.feature_generation.get_features_Sbert(classes[0])
        label2=self.feature_generation.get_features_Sbert(classes[1])
        if self.cosine_similarity(label1,sentence_embedding)>self.cosine_similarity(label2,sentence_embedding):
            return classes[0]
        else:
            return classes[1]


    def run_model_test_creation(self, all_x,all_y,path_static,path_static_core):
        """
        create csv file of human test
        :return:
        """

        X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.8, random_state=42,shuffle=True)
        self.core_on_demand = self.coreOnDemand.create_core_on_demand(X_train,len(set(y_train)),
                                                                      y_train,max_size=768)
        self.core_on_demand = [c.replace("Category:", START_SEQ).replace("_", " ") for c in self.core_on_demand if
                               'Wikipedia categories named after' not in c]
        self.feature_generation = FeatureGeneration(self.core_on_demand)
        classes = []
        for y in y_train:
            if y not in classes:
                classes.append(y)
        assert len(classes)==2

        # us
        self.core_us=self.core_on_demand
        self.embedding_us=self.feature_generation.get_features_explainable_sbert
        point_embedding_us = self.create_all_points_embedding(X_train, y_train, self.embedding_us)
        c_information_gain_c=self.information_gain_all_features(point_embedding_us,self.core_us)


        #static
        self.embedding_static_icos = self.feature_generation.get_features_Sbert
        wordscore=from_yaml_to_python(path_static_core)
        self.core_static_icos=wordscore
        point_embedding_static_icos = self.create_all_points_embedding(X_train, y_train, self.embedding_static_icos)
        c_information_gain_sc=self.information_gain_all_features(point_embedding_static_icos,self.core_static_icos)
        self.point_embedding_model=point_embedding_static_icos


        self.embedding_static = self.feature_generation.get_features_Sbert
        wordscore = from_yaml_to_python(path_static)
        self.core_static = wordscore
        # point_embedding_static = self.create_all_points_embedding(X_train, y_train, self.embedding_static)
        # c_information_gain_s=self.information_gain_all_features(point_embedding_static,self.core_static)
        # assert c_information_gain_s==c_information_gain_sc

        self.train_model(X_train,y_train)


        real_y_list=[]
        model_y_list=[]
        model_y_list_direct = []
        icos_top=[]
        static_icos_top=[]
        static_top=[]
        for i, sentence in enumerate(X_test):
            y_bert,sim_listicos=self.get_explanation_sentence(sentence, y_test[i],
                                          self.embedding_us,self.core_us,c_information_gain_c)

            y_bert, sim_listicosstatic = self.get_explanation_sentence(sentence, y_test[i],
                                                                    self.embedding_static_icos, self.core_static_icos,c_information_gain_sc)
            y_bert, sim_liststatic= self.get_explanation_sentence(sentence, y_test[i],
                                                                    self.embedding_static, self.core_static,c_information_gain_sc)

            y_bert_direct=self.get_prediction(self.feature_generation.get_features_Sbert(sentence),classes)
            # y_bert_direct = self.get_prediction_by_lalel_concept_similarity(self.feature_generation.get_features_Sbert(sentence), classes)
            real_y_list.append(y_test[i])
            model_y_list.append(y_bert)
            model_y_list_direct.append(y_bert_direct)
            icos_top.append(sim_listicos)
            static_icos_top.append(sim_listicosstatic)
            static_top.append(sim_liststatic)
        model_mistake_indexes=[i for i,y in enumerate(real_y_list) if y!=model_y_list[i]]
        model_correct_indexes=[i for i,y in enumerate(real_y_list) if y==model_y_list[i]]
        print(f"number of model mistakes is {len(model_mistake_indexes)} and correct {len(model_correct_indexes)}",flush=True)
        samples_final = []
        self.random_seed()
        samples_final += random.sample(model_mistake_indexes, min(10,len(model_mistake_indexes)))
        self.random_seed()
        samples_final += random.sample(model_correct_indexes, 10)
        real_y_list=[real_y_list[i] for i in samples_final]
        model_y_list=[model_y_list[i] for i in samples_final]
        model_y_list_direct=[model_y_list_direct[i] for i in samples_final]
        icos_top=[icos_top[i] for i in samples_final]
        static_icos_top=[static_icos_top[i] for i in samples_final]
        static_top=[static_top[i] for i in samples_final]
        models=[self.model_check_model.model_sbert, self.model_check_model.model_st5, self.model_check_model.model_distil]
        print(f"number of samples:{len(real_y_list)}")
        print(f"{icos_top=}\n"
              f"{classes=}\n"
              f"{model_y_list=}\n"
              f"{model_y_list_direct=}\n"
              f"{real_y_list=}",flush=True)
        for model in models:
            icos_score=self.model_check_model.get_score_model_directly(classes, icos_top, model)
            static_icos_score=self.model_check_model.get_score_model_directly(classes, static_icos_top, model)
            static_score=self.model_check_model.get_score_model_directly(classes, static_top, model)
            print("NC results")
            print(f"Results of model {model=}\nC*score={self.kapper_test(icos_score,model_y_list_direct)}"
                  f"\nStaticC3score={self.kapper_test(static_icos_score,model_y_list_direct)}"
                  f"\nstatic={self.kapper_test(static_score,model_y_list_direct)}\n",flush=True)

            print("RF results")
            print(f"Results of model {model=}\nC*score={self.kapper_test(icos_score, model_y_list)}"
                  f"\nStaticC3score={self.kapper_test(static_icos_score, model_y_list)}"
                  f"\nstatic={self.kapper_test(static_score, model_y_list)}\n", flush=True)
                  # f"{icos_score=}",flush=True)
