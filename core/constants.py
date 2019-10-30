import os

class Constants:
    FILE_NAME = 'WomensClothing-E-Commerce-Reviews.csv'
    REVISED_FILE_NAME_SENT = 'revised_for_sentiment.csv'
    REVISED_FILE_NAME_ADJ = 'revised_for_adjective.csv'
    COLLECTION_FILE = 'collection.txt'
    RESOURCE_PATH = base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                                      'resources')
