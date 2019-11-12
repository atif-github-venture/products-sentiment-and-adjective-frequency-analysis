<b>Data source<b>:<br>
https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

### What’s this utility?

Browsing through the product can be cumbersome and mostly going through the product reviews to see how it fits your suit. There are many companies adopting to ask for customer reviews where customer is asked to provide specific set of stars/rate for fitting, recommendation and overall satisfaction. But along with this they also leave a bunch of textual information which sits waiting for the other customers to absorb.

There are potential reviews and emotions which deeply buried inside text information which if extracted can enable the customer to make faster decisions. This can also be put to use to create a business intelligence which gives insight to product owners and company executives to really interpret the customer side in an effortless manner.

### Dataset

I started the exploration of various dataset provide over Kaggle. There are lots of publicly available dataset and I chose the Women’s clothing review as it is closer to what I wanted to study.  



Per Wikipedia, Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.


In above used data set, there are mentions of the ratings against each review comment (have filtered the data which does not have comment). To give emotional quotient I have categorized the rating as positive (>3), negative (<3) and neutral (3) as column “sentiment key”, which is saved as “revised file”.


Collection corpus is a collection of data which contains all the words and its frequency in the entire corpus documents. This is used as split to train the classifier using “train set” and then used “test set” to run the classifier to provided sentiment analysis.

To create the collection corpus, have accumulated all the words in the entire document and set the string to “metapy” and then processed each word token to lowercase, removed the “stop words” and applied stemmer (for the root). Then used “nltk word frequency generator” to compute the frequency distribution of each word which is later saved to a “collection.txt” file.












“Collection corpus” is read using “pandas” library for 2 columns namely “word” and “freq” only for first 2000 entries (this collection was sorted before saving) as “word_features”.

The “revised corpus” is read similarly and the 2 columns namely “review text” and “sentiment key” is saved respectively as “document_review” and “document_sent”
“document_review” is then processed for tokenization, lowercase, stop words and stemmer per document (i.e., review text), the size of this data is limited to 1000 as “trainX”.

### Training Naïve Byes Classifier

Each word in “word_features” is checked for existence in each of “document_review” processed set of tokens, if true, feature set for that word is created as “contains({“word})=true/false”, later saved as “featuresets” along with the “sentiment”.

### To Test the classifier for sentiment analysis

Use “test.csv” to provide the text to be analyzed along with the category of sentiment as “neutral/positive/negative”. Also uncomment the “test” section when executing in “sentiment_analyzer.py”

### Adjective classifier

Data files for adjective (revised_adj_file) is loaded. This data set contains reviews bifurcated into “positive” and “negative” classification. For each review text the adjectives from the sentence is extracted as (word_tagged[1] == "JJ" or word_tagged[1] == "JJR" or word_tagged[1] == "JJS"])
Transformed list of sentences (only adjectives) and the existing label is set as training and test dataset.

#### Feature Extraction

The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
•	An instance of the CountVectorizer class is created
•	fit() function is called in order to learn a vocabulary from one or more documents.
•	Then transform() function on one or more documents as needed to encode each as a vector.
An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document.

The vectors returned from a call to transform() will be sparse vectors (Because these vectors will contain a lot of zeros), and can be transformed back to numpy arrays to look and better understand what is going on by calling the toarray() function (X_train.toarray()).

#### Multinomial Naïve Bayes

The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts.

Fit Naive Bayes classifier according to X, y

self.model.fit(X_train, y_train)

Perform classification on an array of test vectors X.

self.model.predict(X_test)


The mean accuracy on the given test data and labels.

self.model.score(X_test, y_test)

#### Confusion matrix

A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.
•	true positives (TP): These are cases in which we predicted Positive, and they are Positive.
•	true negatives (TN): We predicted Negative, and they are Negative.
•	false positives (FP): We predicted Positive, but they are Negative
•	false negatives (FN): We predicted Negative, but they Positive

### Search products for top adjectives

Based on the top adjectives from adjective collection corpus, the entire documents of product are searched and scored in order. This data corpus here is aggregated adjectives of entire review comments summed for a product. 

A metapy library is used which pretty much does heavy lifting on creating inverted document index. All I had to do is run query for terms I am interested in


### Steps to follow for execution

4 main files implemented to provide above solutions.

1.	Process_data.py
2.	Sentiment_analyzer.py
3.	Adjective_classifier.py
4.	Search_top_products.py

Execute in the order above. Please make sure the required libraries are installed for the python SDK you choose. Carefully observe the original data set and its column headers “WomensClothing-E-Commerce-Reviews”.

All the output should be generated and saved in “output” directory.
You can either run throw any IDE or CLI. Just call the main class. Its not designed to be parametrized. 

