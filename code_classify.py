
# coding: utf-8

# In[1]:


#Required libraries

import warnings
warnings.filterwarnings('ignore')
import re
import pickle
import numpy as np
import pandas as pd
import pickle
import operator
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[2]:


def main():
    '''
    Function_description: This function reads the pre-processed train_paras.txt file
    and tags.pkl file.
    
    Input_to_function: None, takes no inputs
    
    Return_values_from_function: does not return any values
        
    dependencies: calls code_classification()
    '''
    
    #Read pre-processed train_paras.txt file into codes
    with open('train_paras.txt','r') as f:
        codes=f.readlines()
    
    for i in range(0,len(codes)):
        codes[i]=codes[i][2:].split("', '")
        codes[i][-1]=codes[i][-1].replace("']\n",'')
    
    #Read tags.pkl as tags
    with open("tags.pkl", "rb") as fp:
        tags = pickle.load(fp)
        
    for i in range(0,len(tags)):
        tags[i]=tags[i][0]
        
    for i in range(0,len(codes)):
        codes[i]=" ".join(word for word in codes[i])
    
    #Call code_classification
    code_classification(codes,tags)
    


# In[21]:


def code_classification(codes,tags):
    '''
    Function_description: This function separates the 20 most frequent tags from
    the data and builds two models to classify the code to respective programming
    language. One model classifies top 9 frequent programming languages and another
    classifies 9-20 most frequent programming languages. Two separate models are
    built to maintain class balance.
    
    Input_to_function: takes codes and tags
    
    Return_values_from_function: none
        
    dependencies: calls model_building()
    '''
    
    #Top 9 most frequent tags
    new_tags1=['javascript','java','php','c#','python','c++','ios','html','sql']
    #Next 11 most frequent tags
    new_tags2=['ruby-on-rails','c','node.js','asp.net','angularjs','excel','iphone','linux'
           'regex','git','scala']
    
    #Indices of specified frequent tags in new_tags1, new_tags2
    indexes_1=[]
    indexes_2=[]
    for i in range(len(tags)):
        if tags[i] in new_tags1:
            indexes_1.append(i)
        elif tags[i] in new_tags2:
            indexes_2.append(i)
            
    #Separating frequent tags to perform classification to maintain class balance
    new_codes1=operator.itemgetter(*indexes_1)(codes)
    tags1=operator.itemgetter(*indexes_1)(tags)
    
    #Model building for top 9 frequent tags
    vectorizer,model=model_building(new_codes1,tags1)
    
    #Saving the vectorizer and model
    with open('vectorizer_1.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model_codes_1.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    #Separating next most frequent tags to perform classification to maintain class balance
    new_codes2=operator.itemgetter(*indexes_2)(codes)
    tags2=operator.itemgetter(*indexes_2)(tags)
    
    #Building model for next 11 frequent tags
    vectorizer,model=model_building(new_codes2,tags2)
    
    #Saving model and vectorizer
    with open('vectorizer_2.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model_codes_2.pkl', 'wb') as f:
        pickle.dump(model, f)
    


# In[22]:


def model_building(new_codes,tags):
    '''
    Function_description: This function builds TF-IDF vectorizer on new_codes
    and the features extracted from TF-IDF are given as input to Logistic Regression
    model.
    
    Input_to_function: takes codes and tags
    
    Return_values_from_function: vectorizer and model
        
    dependencies: None
    '''
    
    #Building TF-IDF
    vectorizer=TfidfVectorizer(analyzer='word',stop_words='english')
    transformed=vectorizer.fit_transform(new_codes)
    
    #Building Logistic regression to classify codes
    model=LogisticRegression()
    model.fit(transformed,tags)
    
    #Return model and vectorizer
    return vectorizer, model


# In[ ]:


if __name__=='__main__':
    main()

