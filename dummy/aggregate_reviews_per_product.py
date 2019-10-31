# import datetime
# import os
# import pandas
#
# st_time = datetime.datetime.now()
# base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
#                                   'resources')
# revised_file = 'WomensClothing-E-Commerce-Reviews-revised.csv'
# agg_file = 'aggregated_comment_per_product.csv'
#
# names = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND', 'Positive Feedback Count',
#          'Division Name', 'Department Name', 'Class Name', 'sentiments key']
# dataset = pandas.read_csv(base_resource_path + '/' + revised_file, names=names)
#
# # get unique set of product ids
# clothing_id = []
# for i, row in dataset.iterrows():
#     clothing_id.append(row['Clothing ID'])
# c_id = list(set(clothing_id))
#
# aggregated_reviews = []
# for id in c_id:
#     filter = dataset['Clothing ID'] == id
#     query_set = dataset.where(filter, inplace=False)
#     query_set = query_set.dropna(subset=['Review Text'])
#     review_set = ''
#     for i, row in query_set.iterrows():
#         review_set = review_set + ' ' +row['Review Text']
#     review_set = review_set.replace(',', ' ').replace('"', "").replace(':', ' ').replace('.', ' ').replace('\'', ' ').replace('\n', ' ').replace('\r', ' ')
#     aggregated_reviews.append(dict({id: review_set}))
#
# with open(base_resource_path + '/' + agg_file, 'w') as fp:
#     for item in aggregated_reviews:
#         for k, v in item.items():
#             fp.write(u'{},{}\n'.format(k, v))
# fp.close()
#
# en_time = datetime.datetime.now()
# print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
