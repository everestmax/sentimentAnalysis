# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 23:06:20 2021

@author: Max
"""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

test = ['Hello im phoning from barclays bank']

with open('RandomForrest.Model', 'rb') as f:
    rf = pickle.load(f)


preds = rf.predict(test)