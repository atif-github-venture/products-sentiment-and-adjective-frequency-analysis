import datetime
import pandas
from resources import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import json

from resources.constants import Constants


class AdjectiveClassifier:
    base_resource_path = Constants.RESOURCE_PATH
    output_path = Constants.OUTPUT_PATH
    revised_adj_file = Constants.REVISED_FILE_NAME_ADJ
    agg_adj_freq_file = Constants.ADJECTIVE_FREQ_DISTRO_PER_PRODUCT_FILE
    class_agg_adj_freq_file = Constants.ADJECTIVE_FREQ_DISTRO_PER_PRODUCT_CLASSIFIED_FILE
    adj_freq_file = Constants.ADJECTIVE_FREQUENCY_FILE
    adj_freq_file_classified = Constants.ADJECTIVE_FREQUENCY_CLASSIFIED_FILE
    top_pos_neg_adj = Constants.TOP_POS_NEG_ADJ

    def __init__(self, picn, hc, recn, rcn, scn):
        self.product_id_col_name = picn
        self.header_columns = hc
        self.review_col_name = recn
        self.rating_col_name = rcn
        self.sentiment_col_name = scn
        self.cv = None
        self.model = None
        self.top = 0

    def extract_list_adject_to_classify(self, data):
        list = []
        for item_list in data:
            for key, val in item_list.items():
                for x in val:
                    for k, v in x.items():
                        list.append(k)
        return list

    def classify_adjective_from_aggr(self, data):
        list_a = []
        for item_list in data:
            for key, val in item_list.items():
                list_dict = []
                for x in val:
                    for k, v in x.items():
                        list_dict.append(dict({'word': k, 'freq': v, 'class': self.classifier(k)[0]}))
                list_a.append(dict({key: list_dict}))
        return list_a

    def adjective_classification(self):
        '''
        Select the data for adjective classification per product.
        Split the data set for train and test
        Use Multinomial Naive Bayes model to fit, score and predict the train/test set
        Print tghe confusion matrix and other metrics and set the model, classifier to class variables respectively for further use.
        :return: None
        '''
        # dataset = pandas.read_csv(base_resource_path + '/' + revised_file, nrows=500, names=names)
        dataset = pandas.read_csv(self.output_path + '/' + self.revised_adj_file, names=self.header_columns)

        X = dataset[self.review_col_name]

        texts_transformed = []
        for review in X:
            adjectives = utils.extract_adjectives(review)
            texts_transformed.append(" ".join(adjectives))

        X = texts_transformed
        y = dataset[self.sentiment_col_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        self.cv = CountVectorizer()
        # cv = CountVectorizer(max_features=500)
        self.cv.fit(X_train)

        X_train = self.cv.transform(X_train)
        X_test = self.cv.transform(X_test)

        arr = X_train.toarray()

        print(arr.shape)

        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

        print(self.model.score(X_test, y_test))

        y_test_pred = self.model.predict(X_test)
        print(confusion_matrix(y_test, y_test_pred))
        print(accuracy_score(y_test, y_test_pred))

    def classifier(self, adjective):
        return self.model.predict(self.cv.transform([adjective]))

    def classify_adjectives_per_product(self):
        '''
        Get the adjective per product and classify and save it back
        :return: None
        '''
        data = utils.read_json_file(self.output_path, self.agg_adj_freq_file)
        classified_list = ac.classify_adjective_from_aggr(data)
        utils.write_to_json(self.output_path, self.class_agg_adj_freq_file, classified_list)

    def classify_adjectives_collection(self):
        '''
        Categorize the collection adjectives for positive/negative and save the output categorized file
        :return: None
        '''
        # dataset = pandas.read_csv(base_resource_path + '/' + revised_file, nrows=500, names=names)
        content = ''
        dataset = pandas.read_csv(self.output_path + '/' + self.adj_freq_file, names=self.header_columns)
        for ind, row in dataset.iterrows():
            word = row['word']
            freq = row['frequency']
            classification = self.classifier(word)[0]
            content += u'{},{},{}\n'.format(word, freq, classification)
        utils.write_to_file(self.output_path, self.adj_freq_file_classified, content)
        self.save_top_adjectives_for_categories()

    def save_top_adjectives_for_categories(self):
        '''
        Extract top positive/negative adjectives in entire collection and save the file
        :return: None
        '''
        #
        content = ''
        classification = 'class'
        self.header_columns.append(classification)
        dataset = pandas.read_csv(self.output_path + '/' + self.adj_freq_file_classified,
                                  names=self.header_columns)

        df_p = dataset.where(dataset[classification] == 'Positive').head(self.top)
        df_n = dataset.where(dataset[classification] == 'Negative')
        df_n = df_n.dropna(subset=[classification]).head(self.top)
        for ind, row in df_p.iterrows():
            content += u'{},{},{}\n'.format(row[self.header_columns[0]], row[self.header_columns[1]],
                                            row[self.header_columns[2]])
        for ind, row in df_n.iterrows():
            content += u'{},{},{}\n'.format(row[self.header_columns[0]], row[self.header_columns[1]],
                                            row[self.header_columns[2]])
        utils.write_to_file(self.output_path, self.top_pos_neg_adj, content)


if __name__ == '__main__':
    st_time = datetime.datetime.now()
    header_columns = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND',
                      'Positive Feedback Count', 'Division Name', 'Department Name', 'Class Name', 'Sentiments key']
    review_col_name = 'Review Text'
    rating_col_name = 'Rating'
    product_id_col_name = 'Clothing ID'
    sentiment_col_name = 'Sentiments key'
    ac = AdjectiveClassifier(product_id_col_name, header_columns, review_col_name, rating_col_name, sentiment_col_name)
    ac.adjective_classification()
    ac.classify_adjectives_per_product()
    ac.header_columns = ['word', 'frequency']
    ac.top = 10
    ac.classify_adjectives_collection()
    # TODO https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
