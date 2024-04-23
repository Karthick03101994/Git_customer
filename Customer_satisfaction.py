        
import pandas as pd
import numpy as np
import nltk
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df=pd.read_csv("tripadvisor_hotel_reviews.csv")
df['catgory']=df['Rating'].map(lambda x: 0 if x in (1,2,3) else 1)
X=df['Review']
Y=df['catgory']
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,train_size=0.80,random_state=1)
clf=Pipeline([('vector',CountVectorizer()),('Logic',LogisticRegression(max_iter=2000))])
clf.fit(xtrain,ytrain)


def main_function(Text):
    result=clf.predict(['Text'])
    return result



