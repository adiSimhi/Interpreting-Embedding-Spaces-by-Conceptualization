
from typing import Tuple, List
import matplotlib.pyplot as plt
from numpy.linalg import norm
from transformers import BertTokenizer, BertModel, BertConfig, GPT2Model, GPT2Tokenizer, GPT2Config
import numpy as np

from core_on_demand import coreOnDemand


class FullLLM():

    def __init__(self,g_graph,core_categories,model_name ="bert-base-uncased" ,text = None):
        self.model_name=model_name
        # model_name="bert-base-uncased"
        if text is None:
            self.core=[c.replace("Category:","") for c in core_categories[2]]
            print(len(self.core))
        else:
            core_on_demand = coreOnDemand(core_categories, g_graph)
            with open(text) as f:
                self.text = f.read()
            print(self.text)
            self.core = core_on_demand.create_core_on_demand(self.text.split(". ")[:10],max_size=768)
            self.core = [c.replace("Category:", "").replace("_", " ") for c in self.core if
                    'Wikipedia categories named after' not in c]
        if self.model_name=="bert-base-uncased":
            config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name,config=config)
        #

        #gpt2-xl model
        else:
            # self.model_name="gpt2"
            config = GPT2Config.from_pretrained(self.model_name, output_hidden_states=True)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2Model.from_pretrained(self.model_name,config=config)
        self.embedding_core=self.create_embedding_for_core()

    def create_embedding_for_core(self)->dict:
        embedding_core={}
        for c in self.core:
            inputs = self.tokenizer(c, return_tensors="pt")
            outputs = self.model(**inputs)
            embedding_core[c]=outputs[2]
        return embedding_core



    def get_explainable_embedding_with_score(self,sentence:str):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        embedding=outputs[2]
        print(f"{sentence:} {np.shape(outputs[0])} {len(embedding)} {np.shape(embedding[0])}")
        all_layers_embedd = []
        for i in range(len(embedding)):
            embedd = []
            if self.model_name=="bert-base-uncased":
                embedding_s_layer = embedding[i][0][1:-1].mean(0).detach().numpy()
            else:
                embedding_s_layer=embedding[i][0].mean(0).detach().numpy()
            for c,c_embedd in self.embedding_core.items():
                assert len(c_embedd)==len(embedding)
                if self.model_name=="bert-base-uncased":
                    c_embedd_layer = c_embedd[i][0][1:-1].mean(0).detach().numpy()
                else:
                    c_embedd_layer=c_embedd[i][0].mean(0).detach().numpy()
                cosin_sim = np.dot(embedding_s_layer, c_embedd_layer) / (norm(c_embedd_layer) * norm(embedding_s_layer))
                embedd.append(cosin_sim)
            all_layers_embedd.append(embedd)
        top_3_concepts_index = np.array(all_layers_embedd[-1]).argsort()[-3:][::-1]
        # top concept index in the first layer
        bottom_3_concepts_index = np.array(all_layers_embedd[0]).argsort()[-3:][::-1]
        top_3_concepts_index = np.concatenate((top_3_concepts_index, bottom_3_concepts_index))
        top_3_concepts = [self.core[index] for index in top_3_concepts_index]
        print(f"{top_3_concepts=}")
        # plot for top 3 concepts the layers rank
        top_3_concepts_rank = []
        for index in top_3_concepts_index:
                top_3_concepts_rank.append(
                    [sum([1 for j in all_layers_embedd[i] if j > all_layers_embedd[i][index]]) for i in
                     range(len(all_layers_embedd))])
        print(f"{top_3_concepts_rank=} {np.shape(top_3_concepts_rank)}")

        return top_3_concepts_rank,top_3_concepts,all_layers_embedd





    def get_llm_graph_for_input(self,word:str):
        top_concept_rank,concept_names,all_layers_embedd=self.get_explainable_embedding_with_score(word)
        x = [i for i in range(1, 14)]
        for index,concept in enumerate(concept_names):
            plt.plot(x,  [i+1 for i in top_concept_rank[index]], label=concept)
            print(f"{concept=},{top_concept_rank[index]=}")
        plt.legend()
        plt.yscale("log")
        plt.gca().invert_yaxis()
        plt.xlabel("Layers")
        plt.ylabel("Rank")
        plt.savefig(word+f"_llm_layers_model{self.model_name}.pdf", format="pdf")
        plt.close()
