# import datetime
# import os
# import pandas
# import matplotlib.pyplot as plt
#
# st_time = datetime.datetime.now()
# base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
#                                   'resources')
# original_file = 'WomensClothing-E-Commerce-Reviews.csv'
# revised_file = 'WomensClothing-E-Commerce-Reviews-revised.csv'
# collection_file = 'collection.txt'
#
# names = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND', 'Positive Feedback Count',
#          'Division Name', 'Department Name', 'Class Name']
# dataset = pandas.read_csv(base_resource_path + '/' + original_file, names=names)
# dataset = dataset[dataset['Review Text'].notnull()]  # Drop the row which does not have comment.
# print('Shape of the data set:', dataset.shape)
# print(dataset.describe())
# print(dataset.groupby('Rating').size())
# dataset.groupby('Rating').size().plot(kind='bar', colormap='Purples_r').set_ylabel('Count')
# plt.show()
#
# dataset['sentiments key'] = dataset['Rating'].apply(
#     lambda x: 'Positive' if (4 <= int(x) <= 5) else ('Neutral' if int(x) == 3 else 'Negative'))
# dataset.to_csv(path_or_buf=base_resource_path + '/' + revised_file, index=False)
#
# dataset.groupby('sentiments key').size().plot(kind='bar', colormap='copper_r').set_ylabel('Count')
# plt.show()
#
# print('That will be it!!')
# en_time = datetime.datetime.now()
# print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
