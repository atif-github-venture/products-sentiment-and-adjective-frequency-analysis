import datetime
import os
import nltk
import pandas
import metapy
import matplotlib.pyplot as plt


def build_collection(doc):
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.ListFilter(tok, base_resource_path + '/stopwords.txt',
                                      metapy.analyzers.ListFilter.Type.Reject)
    tok = metapy.analyzers.Porter2Filter(tok)
    tok.set_content(doc.content())
    return tok


st_time = datetime.datetime.now()
metapy.log_to_stderr()
base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                  'resources')
revised_file = 'WomensClothing-E-Commerce-Reviews-revised.csv'
collection_file = 'collection.txt'

names = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND', 'Positive Feedback Count',
         'Division Name', 'Department Name', 'Class Name', 'sentiments key']
dataset = pandas.read_csv(base_resource_path + '/' + revised_file, names=names)

X = dataset['Review Text']
# X = X[:1000]
complete_set = ''
for i in X:
    complete_set = complete_set + i

doc = metapy.index.Document()
doc.content(complete_set)
tokens = build_collection(doc)

all_words = nltk.FreqDist(tokens)

with open(base_resource_path + '/' + collection_file, 'w') as fp:
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

print('That will be it!!')
en_time = datetime.datetime.now()
print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
