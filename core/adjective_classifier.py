import datetime
import os
import pandas
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import json


def filter_stopwords(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = ''
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence = filtered_sentence + ' ' + ps.stem(w.lower())
    return filtered_sentence


def filter_custom_stop_words(words):
    stop_words = ['i', '-', 'it', 'the', 'a', 'an', 's', 't', 'm', 'd', 'xl', 'xs', 'other']
    filtered_words = []
    for w in words:
        if w.lower() not in stop_words:
            filtered_words.append(w)
    return filtered_words


def extract_list_adject_to_classify(data):
    list = []
    for item_list in data:
        for key, val in item_list.items():
            for x in val:
                for k, v in x.items():
                    list.append(k)
    return list

def classify_adjective_from_aggr(data):
    list = []
    for item_list in data:
        for key, val in item_list.items():
            list_dict = []
            for x in val:
                for k, v in x.items():
                    list_dict.append(dict({'word':k, 'freq': v, 'class': classifier(k)[0]}))
            list.append(dict({key:list_dict}))
    return list

st_time = datetime.datetime.now()
base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                  'resources')
revised_file = 'WomensClothing-E-Commerce-Reviews-revised.csv'
agg_adj_freq_file = 'aggregated_adjective_per_product_freq.json'
class_agg_adj_freq_file = 'classified_aggregated_adjective_per_product_freq.json'
agg_adj_freq_collection_file = 'aggregated_adjective_freq_collection.txt'
agg_adj_freq_collection_file_classified = 'aggregated_adjective_freq_collection_classified.csv'
top_5_pos_neg_adj = 'top_5_pos_neg_adj.txt'

names = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND', 'Positive Feedback Count',
         'Division Name', 'Department Name', 'Class Name', 'sentiments key']
# dataset = pandas.read_csv(base_resource_path + '/' + revised_file, nrows=500, names=names)
dataset = pandas.read_csv(base_resource_path + '/' + revised_file, names=names)

dataset["sentiments key"] = dataset["sentiments key"].replace('Neutral', 'Negative')

# dataset = dataset.sample(frac=1)

X = dataset['Review Text']

texts_transformed = []
for review in X:
    sentences = nltk.sent_tokenize(review)
    adjectives = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence.lower())
        words = filter_custom_stop_words(words)
        words_tagged = nltk.pos_tag(words)
        adj_add = [adjectives.append(word_tagged[0]) for word_tagged in words_tagged if
                   word_tagged[1] == "JJ" or word_tagged[1] == "JJR" or word_tagged[1] == "JJS"]

    texts_transformed.append(" ".join(adjectives))

X = texts_transformed
y = dataset["sentiments key"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

cv = CountVectorizer()
# cv = CountVectorizer(max_features=500)
cv.fit(X_train)

X_train = cv.transform(X_train)
X_test = cv.transform(X_test)

arr = X_train.toarray()

print(arr.shape)

model = MultinomialNB()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

y_test_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(accuracy_score(y_test, y_test_pred))


def classifier(adjective):
    return model.predict(cv.transform([adjective]))


# print(classifier('great'))
# print(classifier('less'))
# print(classifier('short'))
# print(classifier('bad'))
#
# # TODO https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
#
# with open(base_resource_path + '/' + agg_adj_freq_file) as json_file:
#     data = json.load(json_file)
#     for item in extract_list_adject_to_classify(data):
#         print(item + ': ->')
#         print(classifier(item))
#         print('\n')

with open(base_resource_path + '/' + agg_adj_freq_file) as json_file:
    data = json.load(json_file)
    classified_list = classify_adjective_from_aggr(data)
json_file.close()

with open(base_resource_path + '/' + class_agg_adj_freq_file, 'w') as fp:
    # for item in per_product_freq_dist:
    json.dump(classified_list, fp)
fp.close()

# categorize the collection adjectives for positive/negative
names = ['word', 'frequency']
# dataset = pandas.read_csv(base_resource_path + '/' + revised_file, nrows=500, names=names)
dataset = pandas.read_csv(base_resource_path + '/' + agg_adj_freq_collection_file, names=names)

with open(base_resource_path + '/' + agg_adj_freq_collection_file_classified, 'w') as fp:
    for ind, row in dataset.iterrows():
        word = row['word']
        freq = row['frequency']
        classification = classifier(word)[0]
        fp.write(u'{},{},{}\n'.format(word, freq, classification))
fp.close()

# top 5 positive/negative adjectives in entire collection
names = ['word', 'frequency', 'class']
dataset = pandas.read_csv(base_resource_path + '/' + agg_adj_freq_collection_file_classified, names=names)

df_p = dataset.where(dataset['class'] == 'Positive').head(5)
df_n = dataset.where(dataset['class'] == 'Negative')
df_n = df_n.dropna(subset=['class']).head(5)

with open(base_resource_path + '/' + top_5_pos_neg_adj, 'w') as fp:
    for ind, row in df_p.iterrows():
        fp.write(u'{},{},{}\n'.format(row['word'], row['frequency'], row['class']))
    for ind, row in df_n.iterrows():
        fp.write(u'{},{},{}\n'.format(row['word'], row['frequency'], row['class']))
fp.close()

en_time = datetime.datetime.now()
print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))