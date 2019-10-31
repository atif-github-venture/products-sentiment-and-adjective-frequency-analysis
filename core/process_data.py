import datetime
import nltk
import pandas
import matplotlib.pyplot as plt
from core import utils
from core.constants import Constants
import metapy


class ProcessData:
    base_resource_path = Constants.RESOURCE_PATH
    revised_file_sent = Constants.REVISED_FILE_NAME_SENT
    revised_file_adj = Constants.REVISED_FILE_NAME_ADJ
    collection_file = Constants.COLLECTION_FILE
    agg_comment_per_prod_file = Constants.AGGREGATE_COMMENT_FILE
    dataset = None

    def __init__(self, picn, hc, recn, rcn):
        self.product_id_col_name = picn
        self.header_columns = hc
        self.review_col_name = recn
        self.rating_col_name = rcn

    def aggregate_reviews_per_product(self):
        '''
        Find unique list of product ids
        Aggregate all the review comments per product and save to file
        :return: None
        '''
        # get unique set of document ids
        product_ids = []
        for i, row in self.dataset.iterrows():
            product_ids.append(row[self.product_id_col_name])
        c_id = list(set(product_ids)) for i, row in self.dataset.iterrows()

        aggregated_reviews = []
        for id in c_id:
            filter = self.dataset[self.product_id_col_name] == id
            query_set = self.dataset.where(filter, inplace=False)
            query_set = query_set.dropna(subset=[self.review_col_name])
            review_set = ''
            for i, row in query_set.iterrows():
                review_set = review_set + ' ' + row[self.review_col_name]
            review_set = review_set.replace(',', ' ').replace('"', "").replace(':', ' ').replace('.', ' ').replace('\'',
                                                                                                                   ' ').replace(
                '\n', ' ').replace('\r', ' ')
            aggregated_reviews.append(dict({id: review_set}))

        str = ''
        for item in aggregated_reviews:
            for k, v in item.items():
                str += u'{},{}\n'.format(k, v)
        utils.write_to_file(self.base_resource_path, self.agg_comment_per_prod_file, str)

    def generate_collection_corpus(self):
        '''
        Collect the words from dataset review column, count the frequency of each word (excluding stop words)
        Save to file
        :return: None
        '''
        X = self.dataset[self.review_col_name]
        # X = X[:1000]
        complete_set = ''
        for i in X:
            complete_set = complete_set + i

        doc = metapy.index.Document()
        doc.content(complete_set)
        tokens = utils.build_collection(doc, self.base_resource_path, 'stopwords.txt')

        all_words = nltk.FreqDist(tokens)
        str = ''
        for word, frequency in all_words.most_common():
            str += u'{},{}\n'.format(word, frequency)

        utils.write_to_file(self.base_resource_path, self.collection_file, str)

        df = pandas.DataFrame(all_words.most_common())
        df.columns = ['Word', 'Freq']
        print(df)
        ax = df.plot(legend=True, title='Word frequency distribution')
        ax.set_xlabel('Words', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.show()

    def process_input_data(self):
        ''' Read the dataset using pandas library
        Drop the row which does not have comment.
        describe the data by grouping per rating
        Classify the positive/neutral/negative based on ratings
        For Sentiment Analysis: -> Positive as 4, 5; Neutral as 3; Negative as 1, 2
        For Adjective classification: -> Positive as 4, 5; Negative as 1, 2, 3
        :return: None
        '''
        self.dataset = pandas.read_csv(Constants.RESOURCE_PATH + '/' + Constants.FILE_NAME, names=self.header_columns)
        self.dataset = self.dataset[self.dataset[self.review_col_name].notnull()]
        print('Shape of the data set:', self.dataset.shape)
        print(self.dataset.describe())
        print(self.dataset.groupby(self.rating_col_name).size())
        utils.show_plots(self.dataset, 'Purples_r', self.rating_col_name)

        # Sentiment Analysis
        self.dataset['sentiments key'] = self.dataset[self.rating_col_name].apply(
            lambda x: 'Positive' if (4 <= int(x) <= 5) else ('Neutral' if int(x) == 3 else 'Negative'))
        utils.save_dataset_to_csv(self.dataset, self.base_resource_path, self.revised_file_sent)
        utils.show_plots(self.dataset, 'copper_r', 'sentiments key')

        # Adjective classification
        self.dataset["sentiments key"] = self.dataset["sentiments key"].replace('Neutral', 'Negative')
        utils.save_dataset_to_csv(self.dataset, self.base_resource_path, self.revised_file_adj)
        utils.show_plots(self.dataset, 'Greens_r', 'sentiments key')


if __name__ == '__main__':
    st_time = datetime.datetime.now()
    header_columns = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND',
                      'Positive Feedback Count', 'Division Name', 'Department Name', 'Class Name']
    review_col_name = 'Review Text'
    rating_col_name = 'Rating'
    product_id_col_name = 'Clothing ID'
    pp = ProcessData(product_id_col_name, header_columns, review_col_name, rating_col_name)
    pp.process_input_data()
    pp.generate_collection_corpus()
    pp.aggregate_reviews_per_product()
    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
