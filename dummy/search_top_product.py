import datetime
import os
import pandas
import nltk
from collections import defaultdict


class Index:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.index = defaultdict(list)
        self.documents = {}
        self.__unique_id = 0

    def search(self, word):
        word = word.lower()
        return [self.documents.get(id, None) for id in self.index.get(word)]

    def add(self, doc_id, document):
        self.__unique_id = doc_id
        for token in [t.lower() for t in nltk.word_tokenize(document)]:

            if self.__unique_id not in self.index[token]:
                self.index[token].append(self.__unique_id)

        self.documents[self.__unique_id] = document


st_time = datetime.datetime.now()
base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                  'resources')
agg_file = 'aggregated_comment_per_product.csv'

names = ['Clothing ID', 'adjective list']
# dataset = pandas.read_csv(base_resource_path + '/' + revised_file, nrows=500, names=names)
dataset = pandas.read_csv(base_resource_path + '/' + agg_file, names=names)

index = Index(nltk.word_tokenize)

for ind, row in dataset.iterrows():
    index.add(row['Clothing ID'], row['adjective list'])

print(index.search('disappointed'))

en_time = datetime.datetime.now()
print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
