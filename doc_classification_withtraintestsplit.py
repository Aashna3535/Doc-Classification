# -*- coding: utf-8 -*-
"""Doc_classification_withtraintestsplit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m0Euq8lnl68SwJmKM1WFGxDtRZdXygpq

**Dutch Document classification task**
"""

#from google.colab import files
#uploaded = files.upload()
import pickle
import pandas as pd
df= pd.read_csv("Dutch 7_Classification.csv")
df

df.shape

df.info()

df.describe()

df["keywords"].unique()

df["Unnamed: 3"].unique()

df["Unnamed: 0"].unique()

df.isnull().sum()

#printing records having missing values
df[pd.isnull(df).any(axis=1)]

df["keywords"].fillna(df["keywords"].mode()[0], inplace = True)

df["Unnamed: 3"].fillna("Not Given", inplace = True)

df

df.isnull().sum()

import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

df["keywords"].value_counts().plot(kind='bar')

#remove special characters and punctuation
df['keywords'] = df['keywords'].replace(r'[^A-Za-z0-9 ]+', '')

#remove single letters from text
df["keywords"] = df['keywords'].apply (lambda x: re.sub(r"((?<=^)|(?<= )).((?=$)|(?= ))", '', x).strip())

df["keywords"]

#remove special characters and punctuation
df['content'] = df['content'].replace(r'[^A-Za-z0-9 ]+', '')

#remove single letters from text
df["content"] = df['content'].apply (lambda x: re.sub(r"((?<=^)|(?<= )).((?=$)|(?= ))", '', x).strip())

df["content"]

#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()

#df['keywords'] = le.fit_transform(df['keywords'])

X = df['content']
y = df['keywords']

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('dutch'))
from nltk.stem.snowball import SnowballStemmer

print(" ".join(SnowballStemmer.languages))

len(STOPWORDS)

#from google.colab import files
#uploaded = files.upload()

with open('stopwords-nl.txt', 'r') as readfile:
    stopw = readfile.read()
len(stopw)

corpus = []

# (text) rows to clean 
for i in range(0, len(df["content"])):  
      
    # column : "text", row ith 
    text = re.sub('[^a-zA-Z]', ' ', df['content'][i])  
      
    # convert all cases to lower cases 
    text = text.lower()  
      
    # split to array(default delimiter is " ") 
    text = text.split()  
      
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = SnowballStemmer("english")  
      
   
    # loop for stemming each word 
    # in string array at ith row     
    text = [ps.stem(word) for word in text 
             if not word in stopw]  
                  
    # rejoin all string array elements 
    # to create back into a string 
    text = ' '.join(text)   
      
    # append each string to create 
    # array of clean text  
    corpus.append(text)

df['content'] = corpus
df['content']

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.20, random_state = 88)

vectorizer = CountVectorizer()

X_vect = vectorizer.fit_transform(X_train)

from sklearn.svm import SVC  
clf = SVC(kernel='linear')

clf.fit(X_vect,y_train)

#y_pred = clf.predict(vectorizer.transform(X_test))

#print(accuracy_score(y_test,y_pred))

#c = df['content'][0]
#c

#df.head()

#pred = clf.predict(vectorizer.transform([c]))

#print(pred)


# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))


