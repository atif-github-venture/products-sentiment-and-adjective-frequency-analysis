import datetime
import os
import pandas
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def filter_stopwords(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = ''
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence = filtered_sentence + ' ' + ps.stem(w.lower())
    return filtered_sentence


def extract_adjectives(r_text):
    adjectives = []
    words = nltk.word_tokenize(filter_stopwords(r_text))
    words_tagged = nltk.pos_tag(words)
    adj_add = [adjectives.append(word_tagged[0]) for word_tagged in words_tagged if word_tagged[1] == "JJ"]
    return adjectives


st_time = datetime.datetime.now()
base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                  'resources')
agg_file = 'aggregated_comment_per_product.csv'
agg_adj_file = 'aggregated_adjective_per_product.csv'

names = ['Clothing ID', 'Review Text']
dataset = pandas.read_csv(base_resource_path + '/' + agg_file, names=names)

processed_sentence = []
for ind, row in dataset.iterrows():
    c_id = row['Clothing ID']
    r_text = row['Review Text']
    extracted_adj_list = extract_adjectives(r_text)
    processed_sentence.append(u'{},{},{}\n'.format(c_id, extracted_adj_list, r_text))

with open(base_resource_path + '/' + agg_adj_file, 'w') as fp:
    for item in processed_sentence:
        fp.write(item)
fp.close()

en_time = datetime.datetime.now()
print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
