import argparse
import nltk
import random
import string
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.lm import MLE, Laplace, KneserNeyInterpolated, WittenBellInterpolated, Lidstone, StupidBackoff, Vocabulary
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_data(data):

    # Sentence tokenizing and lowering the case
    data = sent_tokenize(data.lower())
    # for new lines and -
    data = [sentence.replace('\n', ' ').replace('-', ' ') for sentence in data]
    # for punctuations
    data = [re.sub(r'[^A-Za-z0-9- ]+', '', sentence) for sentence in data]
    # for empty spaces
    data = [' '.join(sentence.split()) for sentence in data]
    # word tokenizing
    data = [word_tokenize(sentence) for sentence in data] 
    # for empty lists
    preprocessed_data = [list for list in data if list]

    return preprocessed_data

def train_ngram_model(data, n, smoothing):
    ngrams, vocab = padded_everygram_pipeline(n, data)
    smoothing = smoothing.upper()
    if smoothing == "MLE":
        model = MLE(n)
    elif smoothing == "LAPLACE":
        model = Laplace(n)
    elif smoothing == "SB":
        model = StupidBackoff(order=n, alpha=0.9)
    # elif smoothing == "kNI":
    #     model = KneserNeyInterpolated(n)
    elif smoothing == "LIDSTONE":
        model = Lidstone(order=n, gamma=0.1)
    elif smoothing == "WBI":
        model = WittenBellInterpolated(n)
    else:
        raise ValueError("Invalid smoothing method")

    # vocab = Vocabulary(vocab, unk_cutoff=2)
    model.fit(ngrams, vocab)
    # print(model.vocabulary)
    # len(model.vocab)
    return model

def split_dataset(author, sentences, ratio):
    n = len(sentences)
    # print(f'No. of Sentences in {author}: ', n)
    n1 = int(n * ratio / (ratio + 1))
    indices = random.sample(range(n), n)
    train_sen = [sentences[i] for i in indices[:n1]]
    dev_sen = [sentences[i] for i in indices[n1:]]
    return train_sen, dev_sen

def predict(models, authors, author_name, sentence, ngram_n):
    perplexities = {}
    test_ngrams = list(ngrams(pad_both_ends(sentence, n=ngram_n), n=ngram_n))
    # print(test_ngrams)
    for author in authors:
        perplexities[author] = models[author].perplexity(test_ngrams)

    pred_file_name = min(perplexities, key=perplexities.get)
    pred_author = author_name(pred_file_name)
    return pred_author, min(perplexities.values())

def generate_samples(models, authors, num_samples, ngram_n):
    for i in range(num_samples):
        print(f'--- For sample prompt {i+1} generating---')
        for author in authors:
            sample = models[author].generate(10, random_seed=i)
            ngram_sample = list(ngrams(pad_both_ends(sample, n=ngram_n), n=ngram_n))
            perplexity = models[author].perplexity(ngram_sample)
            print(f" Author : {author} -> {sample}")
            print(f" Perplexity : {perplexity:.1f}" )

def main(args):

    with open(args.authorlist, 'r', encoding='utf-8') as f:
        authors_filenames = f.read().splitlines()

    author_name = lambda authors_filenames : authors_filenames.rsplit(".",1)[0] 
    
    # loading the text
    data = {}
    for author in authors_filenames:
        data[author] = load_data(author)
    if args.test:
        data[args.test] = load_data(args.test)

    # data 
    data = {author:preprocess_data(text) for (author, text) in data.items()}

    #splitting
    train_set, dev_set = {}, {}
    if not args.test:
        print("splitting into training and development datasets")
        for (author, text) in data.items():
            dev_set[author], train_set[author] = split_dataset(author, text, 0.1)

    if args.test:
        for (author, text) in data.items():
            if author != args.test:
                train_set[author] = data[author]
            else:
                dev_set[author] = data[author]


    ngram_n = 2
    # MLE, LAPLACE, SB, LIDSTONE, WBI
    smoothing = "LIDSTONE"
    print("training LMs...(this may take a while) ")
    print(f'ngram is : {ngram_n}, model is : {smoothing}')
    num_samples = 5
    models = {}
    authors = train_set.keys()

    # fitting the model
    for author in authors:
        models[author] = train_ngram_model(train_set[author], ngram_n, smoothing)

    if not args.test: 
        print("-----------------Accuracy part without test flag---------------------")
        for author in dev_set.keys():
            correct, total = 0, 0
            for sentence in dev_set[author]:        
 
                pred_author, min_perplexity = predict(models, authors, author_name, sentence, ngram_n)
                actual_author = author_name(author)
                if pred_author == actual_author:
                    correct += 1
                total += 1
            
            accuracy = correct / total
            print(author_name(author) + f"\t{accuracy*100:.1f}%" + " correct")

    if args.test:
        print("-------------Classification part with test flag and testfile----------------")
        for sentence in dev_set[args.test]:

            pred_author , min_perplexity = predict(models, authors, author_name, sentence, ngram_n)
            print(f"Sentence is : {sentence}")
            print(f"Predicted author is : {pred_author}")
            print(f'perplexity for the above sentence :{min_perplexity}')
            print("--------------------------------------")
        # generating 5 samples given the same prompt using the 4 models
        # print("---------------Generation part of 5 samples for each author-----------------")
        # generate_samples(models, authors , num_samples, ngram_n)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Authorship Classifier')
    parser.add_argument('authorlist')
    parser.add_argument('-test', type=str, help='Test file for classification')
    args = parser.parse_args()
    main(args)
