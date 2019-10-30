import nltk
import pandas
import matplotlib.pyplot as plt
from constants import Constants
import metapy


class PreProcessing:
    base_resource_path = Constants.RESOURCE_PATH
    revised_file_sent = Constants.REVISED_FILE_NAME_SENT
    revised_file_adj = Constants.REVISED_FILE_NAME_ADJ
    collection_file = Constants.COLLECTION_FILE
    dataset = None

    def __init__(self, hc):
        self.header_columns = hc

    def build_collection(self, doc):
        tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
        tok = metapy.analyzers.LowercaseFilter(tok)
        tok = metapy.analyzers.ListFilter(tok, self.base_resource_path + '/stopwords.txt',
                                          metapy.analyzers.ListFilter.Type.Reject)
        tok = metapy.analyzers.Porter2Filter(tok)
        tok.set_content(doc.content())
        return tok

    def generate_collection_corpus(self):
        X = self.dataset['Review Text']
        X = X[:1000]
        complete_set = ''
        for i in X:
            i=i.decode(encoding='UTF-8',errors='strict')
            complete_set = complete_set + i

        doc = metapy.index.Document()
        doc.content(complete_set)
        tokens = self.build_collection(doc)

        all_words = nltk.FreqDist(tokens)

        with open(self.base_resource_path + '/' + self.collection_file, 'w') as fp:
            for word, frequency in all_words.most_common():
                fp.write(u'{},{}\n'.format(word, frequency))
        fp.close()

        df = pandas.DataFrame(all_words.most_common())
        df.columns = ['Word', 'Freq']
        print('*********')
        print(df)
        ax = df.plot(legend=True, title='Word frequency distribution')
        ax.set_xlabel('Words', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.show()

    def show_plots(self, dataset, color, group_by):
        dataset.groupby(group_by).size().plot(kind='bar', colormap=color).set_ylabel('Count')
        plt.show()

    def save_dataset_to_csv(self, dataset, filename):
        dataset.to_csv(path_or_buf=self.base_resource_path + '/' + filename, index=False)

    def process_input_data(self):
        ''' Read the dataset using pandas library
        Drop the row which does not have comment.
        describe the data by grouping per rating
        Classify the positive/neutral/negative based on ratings
        For Sentiment Analysis: -> Positive as 4, 5; Neutral as 3; Negative as 1, 2
        For Adjective classification: -> Positive as 4, 5; Negative as 1, 2, 3
        '''
        self.dataset = pandas.read_csv(Constants.RESOURCE_PATH + '/' + Constants.FILE_NAME, names=self.header_columns)
        self.dataset = self.dataset[self.dataset['Review Text'].notnull()]
        print('Shape of the data set:', self.dataset.shape)
        print(self.dataset.describe())
        print(self.dataset.groupby('Rating').size())
        self.show_plots(self.dataset, 'Purples_r', 'Rating')

        # Sentiment Analysis
        self.dataset['sentiments key'] = self.dataset['Rating'].apply(
            lambda x: 'Positive' if (4 <= int(x) <= 5) else ('Neutral' if int(x) == 3 else 'Negative'))
        self.save_dataset_to_csv(self.dataset, self.revised_file_sent)
        self.show_plots(self.dataset, 'copper_r', 'sentiments key')

        # Adjective classification
        self.dataset["sentiments key"] = self.dataset["sentiments key"].replace('Neutral', 'Negative')
        self.save_dataset_to_csv(self.dataset, self.revised_file_adj)
        self.show_plots(self.dataset, 'Greens_r', 'sentiments key')


if __name__ == '__main__':
    header_columns = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND',
                      'Positive Feedback Count', 'Division Name', 'Department Name', 'Class Name']
    pp = PreProcessing(header_columns)
    pp.process_input_data()
    pp.generate_collection_corpus()
