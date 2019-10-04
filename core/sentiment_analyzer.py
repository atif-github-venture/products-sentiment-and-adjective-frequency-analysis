# Sourse -> https://www.datacamp.com/community/tutorials/simplifying-sentiment-analysis-python
# Another example to learn from -> https://medium.com/@ageitgey/text-classification-is-your-new-secret-weapon-7ca4fad15788
# https://www.kaggle.com/iwasdata/nltk-to-classify-adjectives-as-positive-negative
import collections
import datetime
import nltk
import os
import metapy
import pandas
from nltk.metrics import precision, recall, f_measure
from core.collection_generator import build_collection


def process_review(doc):
    tokens = []
    for d in doc:
        docu = metapy.index.Document()
        docu.content(d)
        tokens.append(build_collection(docu))
    return tokens

st_time = datetime.datetime.now()
metapy.log_to_stderr()
base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                  'resources')
revised_file = 'WomensClothing-E-Commerce-Reviews-revised.csv'
collection_file = 'collection.txt'


# Define the feature extractor
names = ['word', 'freq']
collection = pandas.read_csv(base_resource_path + '/' + collection_file, names=names)
word_features = list(collection['word'])[:2000]

rev_names = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND', 'Positive Feedback Count',
         'Division Name', 'Department Name', 'Class Name', 'sentiments key']
dataset = pandas.read_csv(base_resource_path + '/' + revised_file, names=rev_names)
document_review = dataset['Review Text']
document_sent = dataset['sentiments key']

process_review = process_review(document_review[:1000])
trainX = zip(process_review,document_sent[:1000])


def document_features(document):
    document_words = set(document)
    features = {}
    for w in word_features:
        features['contains({})'.format(w)] = (w in document_words)
    return features


# Train Naive Bayes classifier
featuresets = [(document_features(review), sentiment) for (review, sentiment) in trainX]


train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

csObserved = []
for i, (feats, label) in enumerate(featuresets):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    csObserved.append(observed)
    testsets[observed].add(i)

print("Naive Bayes Classifier accuracy: ",(nltk.classify.accuracy(classifier, test_set)))
print("Precision of Positive:", (precision(refsets['Positive'], testsets['Positive'])))
print("Recall of Positive:", (recall(refsets['Positive'], testsets['Positive'])))
print("Fmeasure of Positive:", (f_measure(refsets['Positive'], testsets['Positive'])))
print("Precision of Neutral:", (precision(refsets['Neutral'], testsets['Neutral'])))
print("Recall of Neutral:", (recall(refsets['Neutral'], testsets['Neutral'])))
print("Fmeasure of Neutral:", (f_measure(refsets['Neutral'], testsets['Neutral'])))
print("Precision of Negative:", (precision(refsets['Negative'], testsets['Negative'])))
print("Recall of Negative:", (recall(refsets['Negative'], testsets['Negative'])))
print("Fmeasure of Negative:", (f_measure(refsets['Negative'], testsets['Negative'])))

print('That will be it!!')
en_time = datetime.datetime.now()
print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))