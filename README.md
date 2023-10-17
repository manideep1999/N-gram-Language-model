# N-gram-Language-model using NLTK and from scratch

Author Datafiles : austen_utf8.txt ,  dickens_utf8.txt, tolstoy_utf8.txt, wilde_utf8.txt
Test file : testfile.txt

All the author data files should be defined in the authorlist. Data is split by 90% for training and 10% for development.

N-gram With NLTK : classifier.py
N-gram Without NLTK : classifier_scratch.py

The default value set in the code is n = 2 (bigram model) using Lidstone smoothing. Change the n values and smoothing name inside the code if you want to run on trigram or use different smoothing models. 

## To run with NLTK on all author models:
* python3 classifier.py authorlist

## To classify the sentences in the test file by the model with NLTK :
* python3 classifier.py authorlist -test testfile.txt

## To run without NLTK on all author models:
* python3 classifier_scratch.py authorlist

## To classify the sentences in the test file by the model without NLTK :
* python3 classifier.py authorlist -test testfile.txt
