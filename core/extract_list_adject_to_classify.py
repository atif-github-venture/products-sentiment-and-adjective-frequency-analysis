import datetime
import os
import pandas

st_time = datetime.datetime.now()
base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                  'resources')
agg_adj_file = 'aggregated_adjective_per_product.csv'

names = ['Clothing ID', 'Freq Dist', 'adj list', 'Review Text']
dataset = pandas.read_csv(base_resource_path + '/' + agg_adj_file, names=names)

for ind, row in dataset.iterrows():
    c_id = row['Clothing ID']
    r_text = row['Review Text']
    adj_list = row['adj list']
    freq_dist = row['Freq Dist']
