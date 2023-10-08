# Interpreting Embedding Spaces by Conceptualization

### Steps to run this code:
1. create an environment with environment.yml in the additional files
2. `cd src`
3. `CUDA_VISIBLE_DEVICES="" python run_tests.py -p ../additional_files/ --human_and_model_evaluation --classification_test --triplets_test --example_creation --model_application --full_llm_explained`

To see help about the arguments run `python run_tests.py -h`
 
You can also generate to a sentence it's top CES concepts using [InterpretingEmbeddingSpacesByConceptualization.ipynb](src/InterpretingEmbeddingOfSentence.ipynb) notebook

### How to cite this work:
```@article{simhi2022interpreting,
  title={Interpreting Embedding Spaces by Conceptualization},
  author={Simhi, Adi and Markovitch, Shaul},
  journal={arXiv preprint arXiv:2209.00445},
  year={2022}
}
```