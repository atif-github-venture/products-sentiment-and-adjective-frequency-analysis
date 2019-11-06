import matplotlib.pyplot as plt
import metapy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import nltk
import os


def save_dataset_to_csv(dataset, path, filename):
    dataset.to_csv(path_or_buf=path + '/' + filename, index=False)


def show_plots(dataset, color, group_by):
    dataset.groupby(group_by).size().plot(kind='bar', colormap=color).set_ylabel('Count')
    plt.show()


def build_collection(doc, path, filename):
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.ListFilter(tok, path + '/' + filename,
                                      metapy.analyzers.ListFilter.Type.Reject)
    tok = metapy.analyzers.Porter2Filter(tok)
    tok.set_content(doc.content())
    return tok


def filter_custom_stop_words(words):
    stop_words = ['i', '-', '/', 'it', 'the', 'a', 'an', 's', 't', 'm', 'd', 'xl', 'xs', 'other']
    filtered_words = []
    for w in words:
        if w.lower() not in stop_words:
            filtered_words.append(w)
    return filtered_words


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
    '''
    Extract the adjectives list form text
    :param r_text:
    :return: adjectives: list of adjective for the given string input
    '''
    sentences = nltk.sent_tokenize(r_text)
    adjectives = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence.lower())
        words = filter_custom_stop_words(words)
        words_tagged = nltk.pos_tag(words)
        adj_add = [adjectives.append(word_tagged[0]) for word_tagged in words_tagged if
                   word_tagged[1] == "JJ" or word_tagged[1] == "JJR" or word_tagged[1] == "JJS"]
    return adjectives


def write_to_file(path, filename, content):
    with open(path + '/' + filename, 'w') as fp:
        if isinstance(content, str):
            fp.write(content)
        elif isinstance(content, list):
            for item in content:
                fp.write(item)
    fp.close()


def write_to_json(path, filename, content):
    with open(path + '/' + filename, 'w') as fp:
        json.dump(content, fp)
    fp.close()


def read_json_file(path, filename):
    with open(path + '/' + filename) as json_file:
        data = json.load(json_file)
    json_file.close()
    return data


def transform_with_space(list_a):
    return ' '.join(list_a)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
