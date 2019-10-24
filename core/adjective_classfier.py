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
from sklearn.metrics import confusion_matrix


def filter_stopwords(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = ''
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence = filtered_sentence + ' ' + ps.stem(w.lower())
    return filtered_sentence


st_time = datetime.datetime.now()
base_resource_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                  'resources')
revised_file = 'WomensClothing-E-Commerce-Reviews-revised.csv'
collection_file = 'collection.txt'

names = ['sino', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND', 'Positive Feedback Count',
         'Division Name', 'Department Name', 'Class Name', 'sentiments key']
dataset = pandas.read_csv(base_resource_path + '/' + revised_file, nrows=5000, names=names)

X = dataset['Review Text']

texts_transformed = []
for review in X:
    sentences = nltk.sent_tokenize(review)
    adjectives = []

    for sentence in sentences:
        words = nltk.word_tokenize(filter_stopwords(sentence))
        words_tagged = nltk.pos_tag(words)
        adj_add = [adjectives.append(word_tagged[0]) for word_tagged in words_tagged if word_tagged[1] == "JJ"]

    texts_transformed.append(" ".join(adjectives))

X = texts_transformed
y = dataset["sentiments key"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

cv = CountVectorizer(max_features=50)
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


def classifier(adjective):
    return model.predict(cv.transform([adjective]))


print(classifier('great'))
print(classifier('bad'))

adj = list(zip(model.coef_[0], cv.get_feature_names()))
adj = sorted(adj, reverse=True)
for a in adj:
    print(a)
