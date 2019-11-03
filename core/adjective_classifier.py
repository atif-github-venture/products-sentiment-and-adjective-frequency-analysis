import datetime
import pandas
import nltk
from core import utils
from core.constants import Constants


class AdjectiveClassifier:
    base_resource_path = Constants.RESOURCE_PATH
    agg_reviews_per_prod_file = Constants.AGGREGATE_REVIEWS_FILE
    adj_frequency_file = Constants.ADJECTIVE_FREQUENCY_FILE
    adjectives_per_product_file = Constants.ADJECTIVE_PER_PRODUCT_FILE
    adjectives_freq_dist_per_product_file = Constants.ADJECTIVE_FREQ_DISTRO_PER_PRODUCT_FILE
    classifier = None

    def __init__(self, picn, hc, recn, rcn):
        self.product_id_col_name = picn
        self.header_columns = hc
        self.review_col_name = recn
        self.rating_col_name = rcn


    def aggregate_adjective(self):
        dataset = pandas.read_csv(self.base_resource_path + '/' + self.agg_reviews_per_prod_file,
                                  names=self.header_columns)

        all_adj = []
        processed_sentence = []
        per_product_freq_dist = []
        for ind, row in dataset.iterrows():
            c_id = row[self.product_id_col_name]
            r_text = row[self.review_col_name]
            extracted_adj_list = utils.extract_adjectives(r_text)
            all_adj = all_adj + extracted_adj_list
            all_words = nltk.FreqDist(extracted_adj_list)
            feq_dis = []
            for word, frequency in all_words.most_common():
                feq_dis.append(dict({word: frequency}))
            per_product_freq_dist.append(dict({c_id: feq_dis}))
            processed_sentence.append(u'{} {}\n'.format(c_id, utils.transform_with_space(extracted_adj_list)))

        # Write all the adjective and their respective frequency of entire collection corpus
        str = ''
        for word, freq in nltk.FreqDist(all_adj).most_common():
            str += u'{},{}\n'.format(word, freq)
        utils.write_to_file(self.base_resource_path, self.adj_frequency_file, str)
        # Write adjectives extracted per comment per product
        utils.write_to_file(self.base_resource_path, self.adjectives_per_product_file, processed_sentence)
        # Write adjective frequency distribution per product
        utils.write_to_json(self.base_resource_path, self.adjectives_freq_dist_per_product_file, per_product_freq_dist)

    def extract_list_adject_to_classify(self, data):
        list = []
        for item_list in data:
            for key, val in item_list.items():
                for x in val:
                    for k, v in x.items():
                        list.append(k)
        return list

    def classify_adjective_from_aggr(self, data):
        list = []
        for item_list in data:
            for key, val in item_list.items():
                list_dict = []
                for x in val:
                    for k, v in x.items():
                        list_dict.append(dict({'word':k, 'freq': v, 'class': self.classifier(k)[0]}))
                list.append(dict({key:list_dict}))
        return list

    def classify_adjective_per_review(self):
        pass


if __name__ == '__main__':
    st_time = datetime.datetime.now()
    header_columns = ['Clothing ID', 'Review Text']
    review_col_name = 'Review Text'
    rating_col_name = 'Rating'
    product_id_col_name = 'Clothing ID'
    ac = AdjectiveClassifier(product_id_col_name, header_columns, review_col_name, rating_col_name)
    ac.aggregate_adjective()
    ac.classify_adjective_per_review()
    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
