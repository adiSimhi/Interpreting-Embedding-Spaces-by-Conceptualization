import csv
import logging
import random
from typing import List, Tuple
import numpy as np

from datasets import load_dataset

from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier



from core_on_demand import coreOnDemand

from sklearn.model_selection import StratifiedKFold

from feature_generation import FeatureGeneration
from config import START_SEQ


class ClassificationCheckModelVsInterpretableModel():
    def __init__(self, classification_path,core_all_depth: List[List[str]], graph_g_path):
        self.all_cores = core_all_depth
        self.classification_path = classification_path
        self.coreOnDemand=coreOnDemand(core_all_depth,graph_g_path)
        self.n_split=10
        self.g=graph_g_path


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
        # print(f"the number of distinct x values is {len(x_y)}")
        y_class = set([tuple[1] for tuple in x_y])
        # print(f"the number of classes is {len(y_class)} and they are {y_class}")
        # print(f"average sentence length is {round(sum(sentence_length) / len(sentence_length), 2)}")
        return x_y




    def test_using_random_forest(self, X_train, X_test, y_train, y_test, core_names, func):
        x = [func(x) for x in X_train]
        clf = RandomForestClassifier(max_depth=5, random_state=42)
        clf.fit(x, y_train)
        return [round(clf.score([func(x) for x in X_test], y_test), 2)],list(clf.predict([func(x) for x in X_test]))



    def get_data_20news(self):
        all_data = fetch_20newsgroups(data_home="../additional_files/", subset="all")
        xis = all_data.data
        xis = [x.split("\nSubject:")[-1] for x in xis]
        yis = all_data.target
        y_classes = all_data.target_names
        y_real = [y_classes[y] for y in yis]
        c = list(zip(xis, yis))
        self.random_seed()
        random.shuffle(c)
        X_train, y_train = zip(*c)
        print(f"20_news_dataset")
        return X_train[:min(len(X_train),10000)], y_train[:min(len(X_train),10000)]

    def get_dbpedia_data(self):
        train = load_dataset("dbpedia_14")["train"]
        X_train = [word.get("title")+" "+word.get('content') for word in train]
        y_train = [word.get('label') for word in train]
        # print(f"ys = {set(y_train)}")
        # print(f"{y_train[:10]}")
        for y in y_train:
            if y is None:
                print(f"y is None")
        c = list(zip(X_train, y_train))
        self.random_seed()
        random.shuffle(c)
        X_train, y_train = zip(*c)
        print("dbpedia_14 dataset")
        return X_train[:min(len(X_train),10000)], y_train[:min(len(X_train),10000)]


    def get_ag_news_data(self):
        train = load_dataset("ag_news")["train"]
        X_train = [word.get("text") for word in train]
        y_train = [word.get('label') for word in train ]
        # print(f"ys = {set(y_train)}")
        # print(f"{y_train[:10]}")
        for y in y_train:
            if y is None:
                print(f"y is None")
        c = list(zip(X_train, y_train))
        self.random_seed()
        random.shuffle(c)
        X_train, y_train = zip(*c)
        print(f"ag_news dataset")
        return X_train[:min(len(X_train),10000)], y_train[:min(len(X_train),10000)]


    def get_data_else(self, index):
        data_processed = self.get_data(index)
        self.random_seed()
        all_x_all_y = random.sample(data_processed, min(len(data_processed), 10000))
        all_x = [x_y[0] for x_y in all_x_all_y]
        all_y = [x_y[1] for x_y in all_x_all_y]
        classes = []
        for y in all_y:
            if y not in classes:
                classes.append(y)
        all_y=[classes.index(y) for y in all_y]
        print(f"{self.classification_path[index]} dataset")
        return all_x, all_y

    def random_seed(self):
        random.seed(42)


    def get_data_from_index(self, all_x: List, all_y: List, indexes_list: List[int]) -> Tuple[List, List]:
        """
        seperate x and y
        :param all_x:
        :param all_y:
        :param indexes_list:
        :return:
        """
        x = [all_x[index] for index in indexes_list]
        y = [all_y[index] for index in indexes_list]
        return x, y




    def kapper_test(self,y1,y2):
        p0=sum([1 for j in range(len(y1)) if y1[j]==y2[j]])/len(y1)
        classes=list(set(y1).union(y2))
        pe=0
        for k in classes:
            pe+=sum([1 for j in range(len(y1)) if y1[j]==k])*sum([1 for j in range(len(y2)) if y2[j]==k])
        pe=pe/len(y1)**2
        k=(p0-pe)/(1-pe)

        return p0,k

    def random_kapper_test(self,y1):
        classes = list(set(y1))
        y2=[]
        for sample in y1:
            y2.append(random.choice(classes))
        p0 = sum([1 for j in range(len(y1)) if y1[j] == y2[j]]) / len(y1)
        return p0


    def check_accuracy_all_models(self):
        skf = StratifiedKFold(n_splits=self.n_split,random_state=42, shuffle=True)
        icos_acc=[]
        model_acc=[]
        p0_andk=[]
        raw_random=[]
        for train_index, test_index in skf.split(self.all_x, self.all_y):
            x_t, y_t = self.get_data_from_index(self.all_x, self.all_y, train_index)
            x_test, y_test = self.get_data_from_index(self.all_x, self.all_y, test_index)
            self.core_on_demand = self.coreOnDemand.create_core_on_demand(x_t,
                                                                          len(set(self.all_y)),
                                                                          y_t,N=1000,max_size=768)
            self.core_on_demand = [c.replace("Category:", START_SEQ).replace("_", " ") for c in self.core_on_demand]
            self.feature_generation = FeatureGeneration(self.core_on_demand)

            icosa,predict_icos=self.test_using_random_forest(x_t,
                                         x_test,
                                         y_t,
                                         y_test,
                                         self.core_on_demand,
                                         self.feature_generation.get_features_explainable_sbert)


            modela,predict_mode = self.test_using_random_forest(x_t,
                                                      x_test,
                                                      y_t,
                                                      y_test,
                                                      self.core_on_demand,
                                                      self.feature_generation.get_features_Sbert)
            icos_acc+=icosa
            model_acc+=modela
            p0_andk.append(self.kapper_test(predict_icos,predict_mode))
            raw_random.append(self.random_kapper_test(predict_mode))
        print(
                     f"p0 is {np.average([i[0] for i in p0_andk])} with s.d {np.std([i[0] for i in p0_andk])}\n"
              f"Coefficient {np.average([i[1] for i in p0_andk])} with s.d {np.std([i[1] for i in p0_andk])}\n"
                     f"random p0 is {np.average(raw_random)} with s.d {np.std(raw_random)}\n"
                     f"icos accuracy is {np.average(icos_acc)} with s.d {np.std(icos_acc)}\n"
                     f"model accuracy is {np.average(model_acc)} with s.d {np.std(model_acc)}\n")


    def check_accuracy_on_librery_dataset(self):
        """
        run classification check on dataset ag_news,20news,dbpedia
        :return:
        """
        self.all_x, self.all_y =self.get_data_20news()
        self.check_accuracy_all_models()
        self.all_x, self.all_y =self.get_ag_news_data()
        self.check_accuracy_all_models()
        self.all_x,self.all_y=self.get_dbpedia_data()
        self.check_accuracy_all_models()




    def run_on_all_datasets(self):
        self.check_accuracy_on_librery_dataset()
        for index, data in enumerate(self.classification_path):
            self.all_x, self.all_y = self.get_data_else(index)
            data_together = list(zip(self.all_x, self.all_y))
            self.random_seed()
            random.shuffle(data_together)
            self.all_x, self.all_y = zip(*data_together)


            self.check_accuracy_all_models()