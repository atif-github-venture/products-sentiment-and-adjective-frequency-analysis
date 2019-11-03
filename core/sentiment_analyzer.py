import collections
import datetime
import nltk
import metapy
import pandas
from nltk.metrics import precision, recall, f_measure
from resources import utils
from resources.constants import Constants


class SentimentAnalyzer:
    word_features = None
    base_resource_path = Constants.RESOURCE_PATH
    output_path = Constants.OUTPUT_PATH
    collection_file = Constants.COLLECTION_FILE
    test_file = Constants.SENT_TEST_FILE
    revised_file = Constants.REVISED_FILE_NAME_SENT
    stop_word_file = Constants.STOP_WORD_FILE

    def __init__(self, hc, recn, skcn):
        self.header_columns = hc
        self.review_col_name = recn
        self.sent_key_col_name = skcn

    def process_review(self, doc):
        tokens = []
        for d in doc:
            docu = metapy.index.Document()
            docu.content(d)
            tokens.append(utils.build_collection(docu, self.base_resource_path, self.stop_word_file))
        return tokens

    def document_features(self, document):
        document_words = set(document)
        features = {}
        for w in self.word_features:
            features['contains({})'.format(w)] = (w in document_words)
        return features

    def set_data(self):
        # Define the feature extractor
        names = ['word', 'freq']
        collection = pandas.read_csv(self.output_path + '/' + self.collection_file, names=names)
        self.word_features = list(collection['word'])[:2000]

        dataset = pandas.read_csv(self.output_path + '/' + self.revised_file, names=self.header_columns)
        document_review = dataset[self.review_col_name]
        document_sent = dataset[self.sent_key_col_name]

        processed_review = self.process_review(document_review[:1000])
        self.trainX = zip(processed_review, document_sent[:1000])

    def train_naive_bayes_classifier(self):
        # Train Naive Bayes classifier
        featuresets = [(self.document_features(review), sentiment) for (review, sentiment) in self.trainX]

        train_set, test_set = featuresets[100:], featuresets[:100]
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for index, (text, senti) in enumerate(featuresets):
            refsets[senti].add(index)
            actual = self.classifier.classify(text)
            testsets[actual].add(index)

        print("Naive Bayes Classifier accuracy: ", (nltk.classify.accuracy(self.classifier, test_set)))
        print("Precision of Positive:", (precision(refsets['Positive'], testsets['Positive'])))
        print("Recall of Positive:", (recall(refsets['Positive'], testsets['Positive'])))
        print("Fmeasure of Positive:", (f_measure(refsets['Positive'], testsets['Positive'])))
        print("Precision of Neutral:", (precision(refsets['Neutral'], testsets['Neutral'])))
        print("Recall of Neutral:", (recall(refsets['Neutral'], testsets['Neutral'])))
        print("Fmeasure of Neutral:", (f_measure(refsets['Neutral'], testsets['Neutral'])))
        print("Precision of Negative:", (precision(refsets['Negative'], testsets['Negative'])))
        print("Recall of Negative:", (recall(refsets['Negative'], testsets['Negative'])))
        print("Fmeasure of Negative:", (f_measure(refsets['Negative'], testsets['Negative'])))

    def run_test(self):

        print('****************Testing: START**************')
        c_name = [self.review_col_name, self.sent_key_col_name]
        dataframe = pandas.read_csv(self.base_resource_path + '/' + self.test_file, names=c_name)
        test_review_text = dataframe[c_name[0]]
        test_sentiment = dataframe[c_name[1]]
        p_review = self.process_review(test_review_text[:1])
        doc_test = set(p_review[0])
        feat = {}
        for w in self.word_features:
            feat['contains({})'.format(w)] = (w in doc_test)
        test = self.classifier.classify(feat)
        print('User judgement: ' + test_sentiment[:1])
        print('Classifier prediction: ' + test)
        print('****************Testing: END**************')


if __name__ == '__main__':
    st_time = datetime.datetime.now()
    header_columns = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND',
                      'Positive Feedback Count',
                      'Division Name', 'Department Name', 'Class Name', 'sentiments key']
    review_col_name = 'Review Text'
    sent_key_col_name = 'sentiments key'

    sa = SentimentAnalyzer(header_columns, review_col_name, sent_key_col_name)
    sa.set_data()
    sa.train_naive_bayes_classifier()
    sa.run_test() #To run a test for above classifier uncomment below lines of code.

    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
