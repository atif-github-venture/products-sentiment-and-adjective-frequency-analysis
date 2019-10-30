import datetime
import os
import metapy



st_time = datetime.datetime.now()
base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                  'resources')

idx = metapy.index.make_inverted_index('/Users/aahmed/Documents/FE_GIT/products-sentiment-and-adjective-frequency-analysis/resources/config.toml')


print('abc')

# docno = idx.metadata(doc[0]).get('name')