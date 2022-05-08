#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv("spam_or_not_spam.csv")
print(data.info())

##Here we "vectorize" our emails (basically one-hot encoding) so we can use them
vectorizer = CountVectorizer()
data["email"] = data["email"].replace(np.nan, "error")
vectorized = vectorizer.fit_transform(data["email"])
emailArray = vectorized.toarray()

#Run this line of code to see all the "words" we have in our emails, warning it is very long
#print(vectorizer.get_feature_names())


# In[3]:


#Here we split our dataset into training, eval, and testing sets
trainEmails, testEmails, trainLabels, testLabels = train_test_split(emailArray, data["label"], test_size=0.25)
trainEmails, evalEmails, trainLabels, evalLabels = train_test_split(trainEmails, trainLabels, test_size=0.25)


# In[4]:


baseRDFModel = RandomForestClassifier()
baseRDFModel.fit(trainEmails, trainLabels)
resultsTrain = baseRDFModel.predict(trainEmails)
print("Training Confusion Matrix:")
print(confusion_matrix(trainLabels, resultsTrain))
resultsEval = baseRDFModel.predict(evalEmails) 
print(accuracy_score(trainLabels, baseRDFModel.predict(trainEmails)))
print("Evaluation Confusion Matrix:")
print(confusion_matrix(evalLabels, resultsEval))
resultsTest = baseRDFModel.predict(testEmails) 
print(accuracy_score(evalLabels, baseRDFModel.predict(evalEmails)))
print("Testing Confusion Matrix:")
print(confusion_matrix(testLabels, resultsTest))
print(accuracy_score(testLabels, baseRDFModel.predict(testEmails)))


print("\n\nSVM VERSION")
baseRDFModel = svm.SVC()
baseRDFModel.fit(trainEmails, trainLabels)
resultsTrain = baseRDFModel.predict(trainEmails)
print("Training Confusion Matrix:")
print(confusion_matrix(trainLabels, resultsTrain))
resultsEval = baseRDFModel.predict(evalEmails) 
print(accuracy_score(trainLabels, baseRDFModel.predict(trainEmails)))
print("Evaluation Confusion Matrix:")
print(confusion_matrix(evalLabels, resultsEval))
resultsTest = baseRDFModel.predict(testEmails) 
print(accuracy_score(evalLabels, baseRDFModel.predict(evalEmails)))
print("Testing Confusion Matrix:")
print(confusion_matrix(testLabels, resultsTest))
print(accuracy_score(testLabels, baseRDFModel.predict(testEmails)))


# In[ ]:


##Here we go again, this time removing stop words
data = pd.read_csv("spam_or_not_spam.csv")

vectorizer = CountVectorizer()
stopList = ["the", "to", "a", "and", "in", "is", "be", "which", "on"]
data["email"] = data["email"].replace(np.nan, "error")
for x in data["email"]:
    for y in x:
        if y in stopList:
            x = x.replace(y, "")
vectorized = vectorizer.fit_transform(data["email"])
emailArray = vectorized.toarray()

trainEmails, testEmails, trainLabels, testLabels = train_test_split(emailArray, data["label"], test_size=0.25)
trainEmails, evalEmails, trainLabels, evalLabels = train_test_split(trainEmails, trainLabels, test_size=0.25)

noStopsRDFModel = RandomForestClassifier()
noStopsRDFModel.fit(trainEmails, trainLabels)
resultsTrain = noStopsRDFModel.predict(trainEmails)
print("Training Confusion Matrix:")
print(confusion_matrix(trainLabels, resultsTrain))
resultsEval = noStopsRDFModel.predict(evalEmails) 
print(accuracy_score(trainLabels, noStopsRDFModel.predict(trainEmails)))
print("Evaluation Confusion Matrix:")
print(confusion_matrix(evalLabels, resultsEval))
resultsTest = noStopsRDFModel.predict(testEmails) 
print(accuracy_score(evalLabels, noStopsRDFModel.predict(evalEmails)))
print("Testing Confusion Matrix:")
print(confusion_matrix(testLabels, resultsTest))
print(accuracy_score(testLabels, noStopsRDFModel.predict(testEmails)))

print("\n\nSVM VERSION")
noStopsRDFModel = svm.SVC()
noStopsRDFModel.fit(trainEmails, trainLabels)
resultsTrain = noStopsRDFModel.predict(trainEmails)
print("Training Confusion Matrix:")
print(confusion_matrix(trainLabels, resultsTrain))
resultsEval = noStopsRDFModel.predict(evalEmails) 
print(accuracy_score(trainLabels, noStopsRDFModel.predict(trainEmails)))
print("Evaluation Confusion Matrix:")
print(confusion_matrix(evalLabels, resultsEval))
resultsTest = noStopsRDFModel.predict(testEmails) 
print(accuracy_score(evalLabels, noStopsRDFModel.predict(evalEmails)))
print("Testing Confusion Matrix:")
print(confusion_matrix(testLabels, resultsTest))
print(accuracy_score(testLabels, noStopsRDFModel.predict(testEmails)))


# In[6]:


##Round 3, but we look only at high-risk words
data = pd.read_csv("spam_or_not_spam.csv")

vectorizer = CountVectorizer()
riskyWords = ["free", "opportunity", "investment", "easy", "chance", "link", "nigeria", "save"]
data["email"] = data["email"].replace(np.nan, "error")
for x in data["email"]:
    for y in x:
        if y not in riskyWords:
            x = x.replace(y, "")
vectorized = vectorizer.fit_transform(data["email"])
emailArray = vectorized.toarray()

trainEmails, testEmails, trainLabels, testLabels = train_test_split(emailArray, data["label"], test_size=0.25)
trainEmails, evalEmails, trainLabels, evalLabels = train_test_split(trainEmails, trainLabels, test_size=0.25)

onlyRiskyRDFModel = RandomForestClassifier()
onlyRiskyRDFModel.fit(trainEmails, trainLabels)
resultsTrain = onlyRiskyRDFModel.predict(trainEmails)
print("Training Confusion Matrix:")
print(confusion_matrix(trainLabels, resultsTrain))
resultsEval = onlyRiskyRDFModel.predict(evalEmails) 
print(accuracy_score(trainLabels, onlyRiskyRDFModel.predict(trainEmails)))
print("Evaluation Confusion Matrix:")
print(confusion_matrix(evalLabels, resultsEval))
resultsTest = onlyRiskyRDFModel.predict(testEmails) 
print(accuracy_score(evalLabels, onlyRiskyRDFModel.predict(evalEmails)))
print("Testing Confusion Matrix:")
print(confusion_matrix(testLabels, resultsTest))
print(accuracy_score(testLabels, onlyRiskyRDFModel.predict(testEmails)))


print("\n\nSVM VERSION")
onlyRiskyRDFModel = svm.SVC()
onlyRiskyRDFModel.fit(trainEmails, trainLabels)
resultsTrain = onlyRiskyRDFModel.predict(trainEmails)
print("Training Confusion Matrix:")
print(confusion_matrix(trainLabels, resultsTrain))
resultsEval = onlyRiskyRDFModel.predict(evalEmails) 
print(accuracy_score(trainLabels, onlyRiskyRDFModel.predict(trainEmails)))
print("Evaluation Confusion Matrix:")
print(confusion_matrix(evalLabels, resultsEval))
resultsTest = onlyRiskyRDFModel.predict(testEmails) 
print(accuracy_score(evalLabels, onlyRiskyRDFModel.predict(evalEmails)))
print("Testing Confusion Matrix:")
print(confusion_matrix(testLabels, resultsTest))
print(accuracy_score(testLabels, onlyRiskyRDFModel.predict(testEmails)))

