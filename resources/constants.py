import os


class Constants:
    FILE_NAME = 'WomensClothing-E-Commerce-Reviews.csv'
    REVISED_FILE_NAME_SENT = 'revised_for_sentiment.csv'
    REVISED_FILE_NAME_ADJ = 'revised_for_adjective.csv'
    COLLECTION_FILE = 'collection.csv'
    AGGREGATE_REVIEWS_FILE = 'aggregated_reviews_per_product.csv'
    ADJECTIVE_FREQUENCY_FILE = 'adjective_frequency.csv'
    ADJECTIVE_PER_PRODUCT_FILE = 'adjective_per_product.txt'
    ADJECTIVE_FREQ_DISTRO_PER_PRODUCT_FILE = 'adjective_freq_distro_per_product.json'
    ADJECTIVE_FREQ_DISTRO_PER_PRODUCT_CLASSIFIED_FILE = 'classified_adjective_per_product_freq.json'
    ADJECTIVE_FREQUENCY_CLASSIFIED_FILE = 'adjective_frequency_classified.csv'
    SENT_TEST_FILE = 'sentiment_test_input.csv'
    TOP_POS_NEG_ADJ = 'top_10_pos_neg_adj.csv'
    STOP_WORD_FILE = 'stopwords.txt'
    RESOURCE_PATH = os.path.join(
        os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
        'resources')
    OUTPUT_PATH = os.path.join(
        os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
        'output')
