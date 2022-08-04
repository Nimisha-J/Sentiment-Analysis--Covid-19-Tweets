# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:22:23 2021

@author: Nimisha
"""

import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split

np.random.seed(101)
data=pd.read_csv('C:/Users/Nimis/Desktop/Main project/Code/stepbystep/data.csv', encoding = 'utf8')
data.tweet=data.tweet.astype(str)
print(data.head())


Y = pd.get_dummies(data['sentiment']).values
data['label_convert'] = data['sentiment'].map({'negative':0, 'neutral':1, 'positive': 2})
Y = np.array(data['label_convert'])
print(Y)

from sklearn.feature_extraction.text import CountVectorizer

# This converts the list of words into space-separated strings
#data['tweet'] = data['tweet'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()
counts = count_vect.fit_transform(data['tweet'])
print("count vectorizer",counts)
print(len(count_vect.vocabulary_))
print(count_vect.vocabulary_)
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts)
print("transformer",transformer)
print(counts)

from sklearn.metrics import f1_score, classification_report,confusion_matrix, accuracy_score,precision_score,recall_score
# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(counts,Y, test_size=0.2, random_state=69)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.naive_bayes import MultinomialNB

# modelling using the Multinomial Naive Bayes model
model = MultinomialNB().fit(X_train, y_train)

# predicting using test set
predicted = model.predict(X_test)
print("----Naive Bayes classification report ---- ") 
print(pd.crosstab(y_test.ravel(), predicted, rownames = ['True'], colnames = ['Predicted'], margins = True))       
print("-----------------------------------------")
    #print(classification_report(test_y, test_y_hat))
print("Accuracy -- ")
print(accuracy_score(y_test, predicted))
print('Precision: ' , precision_score(y_test, predicted, average="macro"))
print('Recall: ', recall_score(y_test, predicted, average="macro"))
print('F1 Score:',f1_score(y_test, predicted, average="macro"))
print(np.mean(predicted == y_test))

from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier(n_estimators=5,  random_state=82)
random.fit(X_train, y_train) 
# Prediction 
y_pred = random.predict(X_test)
print("----Random forest classification report ---- ") 
print(pd.crosstab(y_test.ravel(), y_pred, rownames = ['True'], colnames = ['Predicted'], margins = True))       
print("-----------------------------------------")
    #print(classification_report(test_y, test_y_hat))
print("Accuracy -- ")
print(accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred, average="macro"))
print('Recall:', recall_score(y_test, y_pred, average="macro"))
print('F1 Score:', f1_score(y_test, y_pred, average="macro"))
# Evaluation of the model
#print(np.mean(y_pred == y_test))
'''
#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("----SVM classification report ---- ") 
print(pd.crosstab(y_test.ravel(), y_pred, rownames = ['True'], colnames = ['Predicted'], margins = True))       
print("-----------------------------------------")
    #print(classification_report(test_y, test_y_hat))
print("Accuracy -- ")
print(accuracy_score(y_test, y_pred))

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(binarize = 0.1)


model = clf.fit(X_train, y_train)
# Prediction 
y_pred = model.predict(X_test)

# Evaluation of the model
print(np.mean(y_pred == y_test))

# Building the model 
from sklearn.svm import SVC
rbf = SVC(kernel='rbf', gamma=4, C=10)
# Training the model using the training set
rbf.fit(X_train, y_train)

# Prediction 
y_pred = rbf.predict(X_test)
print("----SVM classification report ---- ") 
print(pd.crosstab(y_test.ravel(), y_pred, rownames = ['True'], colnames = ['Predicted'], margins = True))       
print("-----------------------------------------")
    #print(classification_report(test_y, test_y_hat))
print("Accuracy -- ")
print(accuracy_score(y_test, y_pred))'''

# Evaluation of the model
#print(np.mean(y_pred == y_test))