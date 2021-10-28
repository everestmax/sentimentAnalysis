# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
import re
import nltk
from pandas import read_csv
nltk.download('stopwords')
import pickle
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TRAIN_PERCENTAGE = 0.67

filename = 'Data/Transcripts.csv'

data = read_csv(filename, skiprows = lambda x: (x != 0) and not x%2)

stemmer = WordNetLemmatizer()

for sen in data['text']:
     # Remove all the special characters
    document = re.sub(r'\W', ' ', str(sen))
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Lemmatization
    document = document.split()
    
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    data.loc[data['text'] == sen, 'text'] = document

#Tokenize the words and convert them into words 
#max features is the maximum number of unique words
#vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
#Give value to each word deending on how many times they occur in other documents

tfidfconverter = TfidfVectorizer(max_features=1500, min_df=3, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(data['text']).toarray()

for i in range(0, len(X)):
    data.loc[i, 'text'] = X[i]


data_pos = data.loc[data['label'] == 'positive']
data_neg = data.loc[data['label'] == 'negative']

pos_rand_choices = random.sample(range(0, len(data_pos)), round(TRAIN_PERCENTAGE*len(data_pos)))
train_data = data_pos.iloc[pos_rand_choices]
test_data = data_pos.drop(train_data.index, 0)

neg_rand_choices =  random.sample(range(0, len(data_neg)), round(TRAIN_PERCENTAGE*len(data_neg)))
neg_train_data = data_neg.iloc[neg_rand_choices]
neg_test_data = data_neg.drop(neg_train_data.index, 0)
 
train_data = train_data.append(neg_train_data)
test_data = test_data.append(neg_test_data)

x_train = list(train_data['text'])
y_train = list(train_data['label'])

x_test = list(test_data['text'])
y_test = list(test_data['label'])

#fit the data to a randomforrest model
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(x_train, y_train) 

#predict the test data 
y_pred = classifier.predict(x_test)

#evaluate the model
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


with open('RandomForrest.Model', 'wb') as f:
    pickle.dump(classifier, f)

pickle.dump(tfidfconverter.vocabulary_,open("feature.pkl","wb"))


### Test a custom phrase
# =============================================================================
# test = ['Hello im phoning from barclays bank']
# test_vector = tfidfconverter.fit_transform(test).toarray()
# custom_output = classifier.predict(test_vector)
# print(custom_output)
# =============================================================================

