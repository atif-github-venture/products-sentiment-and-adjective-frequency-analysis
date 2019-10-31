import collections
import datetime
import nltk
import os
import metapy
import pandas
from nltk.metrics import precision, recall, f_measure


class SentimentAnalyzer:

    def build_collection(doc):
        tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
        tok = metapy.analyzers.LowercaseFilter(tok)
        tok = metapy.analyzers.ListFilter(tok, base_resource_path + '/stopwords.txt',
                                          metapy.analyzers.ListFilter.Type.Reject)
        tok = metapy.analyzers.Porter2Filter(tok)
        tok.set_content(doc.content())
        return tok


    def process_review(doc):
        tokens = []
        for d in doc:
            docu = metapy.index.Document()
            docu.content(d)
            tokens.append(build_collection(docu))
        return tokens

    def document_features(document):
        document_words = set(document)
        features = {}
        for w in word_features:
            features['contains({})'.format(w)] = (w in document_words)
        return features


if __name__ == '__main__':
    st_time = datetime.datetime.now()
    base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                      'resources')
    revised_file = 'WomensClothing-E-Commerce-Reviews-revised.csv'
    collection_file = 'collection.txt'
    test_file = 'test.csv'

    # Define the feature extractor
    names = ['word', 'freq']
    collection = pandas.read_csv(base_resource_path + '/' + collection_file, names=names)
    word_features = list(collection['word'])[:2000]

    rev_names = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND',
                 'Positive Feedback Count',
                 'Division Name', 'Department Name', 'Class Name', 'sentiments key']
    dataset = pandas.read_csv(base_resource_path + '/' + revised_file, names=rev_names)
    document_review = dataset['Review Text']
    document_sent = dataset['sentiments key']

    processed_review = process_review(document_review[:1000])
    trainX = zip(processed_review, document_sent[:1000])

    # Train Naive Bayes classifier
    featuresets = [(document_features(review), sentiment) for (review, sentiment) in trainX]

    train_set, test_set = featuresets[100:], featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for index, (text, senti) in enumerate(featuresets):
        refsets[senti].add(index)
        actual = classifier.classify(text)
        testsets[actual].add(index)

    print("Naive Bayes Classifier accuracy: ", (nltk.classify.accuracy(classifier, test_set)))
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

    # To run a test for above classifer uncomment below lines of code.
    # print('****************Testing: START**************')
    # c_name = ['text', 'sent']
    # dataframe = pandas.read_csv(base_resource_path + '/' + test_file, names=c_name)
    # test_review_text = dataframe['text']
    # test_sentiment = dataframe['sent']
    # p_review = process_review(test_review_text[:1])
    # doc_test = set(p_review[0])
    # feat = {}
    # for w in word_features:
    #     feat['contains({})'.format(w)] = (w in doc_test)
    # test = classifier.classify(feat)
    # print('User judgement: ' + test_sentiment[:1])
    # print('Classifier prediction: ' + test)
    # print('****************Testing: END**************')
