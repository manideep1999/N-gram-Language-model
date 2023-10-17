import argparse
import random
import string
import re
import math

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_data(data):

    # Sentence tokenizing and lowering the case
    data = re.split(r'[.!?]', data.lower()) 
    # for new lines and -
    data = [sentence.replace('\n', ' ').replace('-', ' ') for sentence in data]
    # for punctuations
    data = [re.sub(r'[^A-Za-z0-9- ]+', '', sentence) for sentence in data]
    # for empty spaces
    data = [' '.join(sentence.split()) for sentence in data]
    # word tokenizing
    data = [sentence.split(' ') for sentence in data]
    # for empty lists
    preprocessed_data = [list for list in data if list]

    return preprocessed_data

def train_ngram_model(data, n, smoothing):
    ngrams = []
    for sentence in data:
        ngrams.extend(zip(*[sentence[i:] for i in range(n)]))
    ngram_counts = dict()
    for ngram in ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    total_ngrams = len(ngrams) 
    if smoothing.upper() == "LIDSTONE":
        model = ngram_counts
    else:
        raise Exception("Wrong model name")
    return model, total_ngrams  

def compute_perplexity(test_data, model):
    return model.perplexity(test_data)

def split_dataset(author, sentences, ratio):
    n = len(sentences)
    print(f'No. of Sentences in {author}: ', n)
    n1 = int(n * ratio / (ratio + 1))
    indices = random.sample(range(n), n)
    train_sen = [sentences[i] for i in indices[:n1]]
    dev_sen = [sentences[i] for i in indices[n1:]]
    return train_sen, dev_sen

def main():

    parser = argparse.ArgumentParser(description='Authorship Classifier')
    parser.add_argument('authorlist')
    parser.add_argument('-test', type=str, help='Test file for classification')
    args = parser.parse_args()

    with open(args.authorlist, 'r') as f:
        author_list = f.read()
    author_names = author_list.splitlines()
    author_name = lambda author_names : author_names.rsplit(".",1)[0] 

    # load the text
    data = {}
    for author in author_names:
        data[author] = load_data(author)
    
    if args.test:
        data[args.test] = load_data(args.test)

    data = {author:preprocess_data(text) for (author, text) in data.items()}

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
    smoothing = "LIDSTONE"
    print("training LMs...(this may take a while) ")
    models = {}
    total_ngrams = 0
    for author in train_set.keys():
        model, ngrams = train_ngram_model(train_set[author], ngram_n, smoothing)
        models[author] = model
        total_ngrams += ngrams

    if not args.test: 
        print("-----------------Accuracy part without test flag---------------------")

        for author in dev_set.keys():
            correct, total = 0, 0

            for sentence in dev_set[author]:
                ngrams = list(zip(*[sentence[i:] for i in range(ngram_n)]))
                # print(ngrams)

                perplexities = {}
                for file_name_train in train_set.keys():
                    model = models[file_name_train]
                    perplexity = 0.0
                    for ngram in ngrams:
                        ngram_count = model.get(ngram, 0)
                        perplexity += -1.0 / len(ngrams) * math.log((ngram_count + 1) / (total_ngrams + len(model)))
                    perplexities[file_name_train] = perplexity
                    # print(perplexities)

                # print("------------------------------")
                pred_file_name = min(perplexities, key=perplexities.get)

                pred_author = author_name(pred_file_name)
                actual_author = author_name(author)

                if pred_author == actual_author:
                    correct += 1
                total += 1
            
            accuracy = correct / total
            print(author_name(author) + f"\t{accuracy*100:.1f}%" + " correct")

    if args.test:
        # print("---------")
        print("-------------Classification part with test flag and testfile----------------")
        for sentence in dev_set[args.test]:
            ngrams = list(zip(*[sentence[i:] for i in range(ngram_n)]))
            # print(ngrams)
            
            perplexities = {}
            for author in train_set.keys():
                model = models[author]
                perplexity = 0.0
                for ngram in ngrams:
                    ngram_count = model.get(ngram, 0)
                    perplexity += -1.0 / len(ngrams) * math.log((ngram_count + 1) / (total_ngrams + len(model)))
                perplexities[author] = perplexity
                # print(perplexities)
            pred_file_name = min(perplexities, key=perplexities.get)
            pred_author = author_name(pred_file_name) 
            print(f"Sentence is : {sentence}")
            print(f"Predicted author is : {pred_author}")
            print(f'perplexity for the above sentence :{min(perplexities.values())}')
            print("--------------------------------------")
    
if __name__ == '__main__':
    main()
