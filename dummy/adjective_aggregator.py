import datetime
import os
import pandas
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json


def filter_stopwords(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = ''
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence = filtered_sentence + ' ' + ps.stem(w.lower())
    return filtered_sentence


def filter_custom_stop_words(words):
    stop_words = ['i', '-', 'it', 'the', 'a', 'an', 's', 't', 'm', 'd', 'xl', 'xs', 'other']
    filtered_words = []
    for w in words:
        if w.lower() not in stop_words:
            filtered_words.append(w)
    return filtered_words


def extract_adjectives(r_text):
    # words = nltk.word_tokenize(filter_stopwords(r_text))
    texts_transformed = []
    sentences = nltk.sent_tokenize(r_text)
    adjectives = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence.lower())
        words = filter_custom_stop_words(words)
        words_tagged = nltk.pos_tag(words)
        adj_add = [adjectives.append(word_tagged[0]) for word_tagged in words_tagged if
                   word_tagged[1] == "JJ" or word_tagged[1] == "JJR" or word_tagged[1] == "JJS"]
    return adjectives


st_time = datetime.datetime.now()
base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                  'resources')
agg_file = 'aggregated_comment_per_product.csv'
agg_adj_file = 'aggregated_adjective_per_product.csv'
agg_adj_freq_file = 'aggregated_adjective_per_product_freq.json'
agg_adj_freq_collection_file = 'aggregated_adjective_freq_collection.txt'

names = ['Clothing ID', 'Review Text']
dataset = pandas.read_csv(base_resource_path + '/' + agg_file, names=names)

all_adj = []
processed_sentence = []
per_product_freq_dist = []
for ind, row in dataset.iterrows():
    c_id = row['Clothing ID']
    r_text = row['Review Text']
    extracted_adj_list = extract_adjectives(r_text)
    all_adj = all_adj + extracted_adj_list
    all_words = nltk.FreqDist(extracted_adj_list)
    feq_dis = []
    for word, frequency in all_words.most_common():
        feq_dis.append(dict({word: frequency}))
    per_product_freq_dist.append(dict({c_id: feq_dis}))
    processed_sentence.append(u'{},{}\n'.format(c_id, extracted_adj_list))


with open(base_resource_path + '/' + agg_adj_freq_collection_file, 'w') as fp:
    for word, freq in nltk.FreqDist(all_adj).most_common():
        fp.write(u'{},{}\n'.format(word, freq))
fp.close()

with open(base_resource_path + '/' + agg_adj_file, 'w') as fp:
    for item in processed_sentence:
        fp.write(item)
fp.close()

with open(base_resource_path + '/' + agg_adj_freq_file, 'w') as fp:
    # for item in per_product_freq_dist:
    json.dump(per_product_freq_dist, fp)
fp.close()

en_time = datetime.datetime.now()
print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
