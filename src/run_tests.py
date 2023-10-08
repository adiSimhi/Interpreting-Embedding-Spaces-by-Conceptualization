import argparse
import os
import config
from classification_test import ClassificationCheckModelVsInterpretableModel
from example_creation import ExamplesOfModel
from model_aplication import checkPhrases
from model_test_model_classification import ModelTest
from full_llm import FullLLM
from using_pickle import from_yaml_to_python
from triplets_test import TripletsTest

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--path_to_data", required=True,help="additional data path")
    parser.add_argument("--human_and_model_evaluation", action="store_true", help="creating the data for the human test and running the evaluation by other models")
    parser.add_argument("--classification_test", action="store_true",help = "run the classification test")
    parser.add_argument("--example_creation", action="store_true",help = "create the three examples of the sentence top features from CNN articles")
    parser.add_argument("--model_application",action="store_true",help = "create the few examples of model application as mentioned in our paper")
    parser.add_argument("--triplets_test", action="store_true", help = "run the triplet test")
    parser.add_argument("--full_llm_explained", action="store_true",help = "create the full LLM layer graphs for GPT2 and BERT on the text 'Government'")
    return parser.parse_args()

def get_data(path_to_data):
    path_to_graph = os.path.join(path_to_data, config.GRAPH_CATEGORIES)
    all_core_categories=[]
    for depth in config.DEPTHS_PATH:
        all_core_categories.append(
            from_yaml_to_python(os.path.join(path_to_data, depth)))
    return path_to_graph,all_core_categories

def classification_test(path_to_data):
    path_to_graph,core_categories_list=get_data(path_to_data)

    classification=ClassificationCheckModelVsInterpretableModel([os.path.join(path_to_data, file) for file in config.NAME_OF_CLASSIFICATION_LIST], core_categories_list, path_to_graph)
    classification.run_on_all_datasets()


def full_llm_results(path_to_data):
    path_to_graph, core_categories_list = get_data(path_to_data)
    llm_explained=FullLLM(path_to_graph, core_categories_list,model_name="gpt2")

    llm_explained.get_llm_graph_for_input("Government")

    llm_explained = FullLLM(path_to_graph, core_categories_list,model_name="bert-base-uncased")

    llm_explained.get_llm_graph_for_input("Government")

def static_vs_icos(path_to_data):
    path_to_graph, core_categories_list = get_data(path_to_data)
    modeltest = ModelTest(
        [os.path.join(path_to_data, file) for file in config.NAME_OF_CLASSIFICATION_LIST], core_categories_list,
        path_to_graph)
    modeltest.run_all_options(os.path.join(path_to_data,config.STATIC_EMBEDDING_DIM[0]),os.path.join(path_to_data,config.STATIC_EMBEDDING_DIM[1]))


def example_creation(path_to_data):
    path_to_graph, core_categories_list = get_data(path_to_data)
    ExamplesOfModel(core_categories_list, path_to_graph, [os.path.join(path_to_data, "corona_different_text.txt")
    , os.path.join(path_to_data, "james_telescope_cnn_text.txt"), os.path.join(path_to_data,
                                                                                   "AI_made_these_stunning_images.txt")])


def model_application(path_to_data):
    path_to_graph, core_categories_list = get_data(path_to_data)
    check_phrases=checkPhrases(core_categories_list[2])
    check_phrases.get_phrase_top(config.PHRASES)
    check_phrases.triple_check(config.PHRASES_TRIPLES)

def triplets_test(path_to_data):
    path_to_graph, core_categories_list = get_data(path_to_data)
    triplets=TripletsTest(os.path.join(path_to_data,"datasetIBMAscii.csv"),core_categories_list)
    triplets.agreement_on_triplets()

def run_all(args):
    path_to_data = args.path_to_data
    if args.example_creation == True:
        print("example_creation:")
        example_creation(path_to_data)

    if args.triplets_test ==True:
        print("triplets_test")
        triplets_test(path_to_data)
    if args.human_and_model_evaluation==True:
        print("human_and_model_evaluation:")
        static_vs_icos(path_to_data)

    if args.model_application==True:
        print("model_application:")
        model_application(path_to_data)
    if args.full_llm_explained==True:
        print("full_llm_explained")
        full_llm_results(path_to_data)
    if args.classification_test == True:
        print("classification_test:")
        classification_test(path_to_data)
    return


def main():
    """
    running all
    :return:
    """
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()