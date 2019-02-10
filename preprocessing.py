
# coding: utf-8

# In[1]:


#Required libraries

import warnings
warnings.filterwarnings('ignore')
import pickle
import re
import gc
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.models import Model
from keras import backend as K
from itertools import chain


# In[17]:


def main():
    '''Function_description: This function is the central function which orchestrates the
    entire program from reading files to preprocessing and saving processed text for furthur
    model building.
    
    Train file consists of three columns : Article, Title, Tag 
    Test file consists of Article and Title.
    
    Input_to_function: None
    
    Output_from_function: None
    
    Dependencies: calls article_preprocessing(), paras_text_processing()
    '''
    
    #Reading train and test data
    train_data=pd.read_csv("train.csv")
    test_data=pd.read_csv("test.csv")
    
    #Combining train and test data for preprocessing
    articles=list(train_data['article'])+list(test_data['article'])
    tags=list(train_data['tags'])
    title=list(train_data['title'])+list(test_data['title'])
    
    #Proprocessing of articles, titles and tags of combined train and test data
    paras,tags=article_preprocessing(articles,titles,tags)
    
    #Text processing on paras extracted from article_preprocessing()
    paras=paras_text_processing(paras)
    
    #Splitting paras on next line
    for i in range(0,len(paras)):
        paras[i]=paras[i].split()
    
    #Separating test data from paras
    test_paras=paras[len(tags):]

    #Saving pre-processed test data into file "test_paras.txt"
    with open("test_paras.txt", "w") as f:
        for para in test_paras:
            f.write(str(para) +"\n")
    
    #Separating train data from paras
    paras=paras[0:len(tags)]
    
    #Saving pre-processed train data into file "train_paras.txt"
    with open("train_paras.txt", "w") as f:
        for para in paras:
            f.write(str(para) +"\n")
            
    #Saving pre-processed tags into "tags.pkl"
    with open('tags.pkl', 'wb') as f:
        pickle.dump(tags, f)
        
    print("Files written to 'train_paras.txt', 'test_paras.txt','tags.pkl'")


# In[19]:


def article_preprocessing(articles,title,tags):
    '''Function_description: This function does html parsing of articles to separate code and
    description contained in the text. It also performes text preprocessing of removing of tags
    from the codes and description. It then combines title, description and codes into paras. It 
    also does preprocessing on tags. It converts '|' separated tags to a list.
    
    Input_to_function: articles, title and tags
    
    Output_from_function: preprocessed paras and tags
    
    Dependencies: None
    '''
    
    #Separating codes and paras using HTML parsing
    codes=[]
    paras=[]
    for i in range(0,len(articles)):
        html = articles[i]
        parsed_html = BeautifulSoup(html,"lxml")
        codes.append(str(parsed_html.findAll('code')))
        paras.append(str(parsed_html.findAll('p')))
    
    #Removing tags in codes
    for i in range(0,len(codes)):
        if len(codes[i])==2:
            codes[i]=""
        else:
            codes[i]=codes[i].replace("\r","")
            codes[i]=codes[i].replace("\n","")
            codes[i]=codes[i].replace('[<code>',"")
            codes[i]=codes[i].replace('</code>]',"")
            codes[i]=codes[i].replace('<code>',"")
            codes[i]=codes[i].replace('</code>',"")
            codes[i]=codes[i].replace("  ","")
        
        #Removing tags in paras
        paras[i]=paras[i].replace("[<p>","")
        paras[i]=paras[i].replace("</p>]","")
        paras[i]=paras[i].replace("<p>","")
        paras[i]=paras[i].replace("</p>","")
        paras[i]=paras[i].replace("<code>","")
        paras[i]=paras[i].replace("</code>","")
        paras[i]=paras[i].replace("\r","")
        paras[i]=paras[i].replace("\n","")
    
    #Combining preprocessed title, paras and codes
    for i in range(0,len(paras)):
        paras[i]=title[i]+" "+paras[i]+" "+codes[i]
        paras[i]=paras[i].lower()
        paras[i]=paras[i].replace("lt","")
        paras[i]=paras[i].replace("gt","")
        paras[i]=paras[i].split()
        
    #Splitting tags on '|' and converting to list
    for i in range(0,len(tags)):
        try:
            tags[i]=tags[i].split('|')
        except:
            tags[i]=['no_tag']
            
    #Returning preprocessed paras and tags      
    return paras,tags
    


# In[25]:


def paras_text_processing(paras):
    '''Function_description: This function does text preprocessing on paras. It removes stop
    words and also removes punctuation, single characters, numbers etc 
    
    Input_to_function: paras
    
    Output_from_function: preprocessed paras
    
    Dependencies: None
    '''
    #NLTK stop words
    stop = set(stopwords.words('english'))
    
    #Removing stop words
    for i in range(0,len(paras)):
        paras[i]=[word for word in paras[i] if word not in stop]
        paras[i]=" ".join(word for word in paras[i])
        
    #Regex to remove punctuation, single characters and numbers
    for i in range(0,len(paras)):
        paras[i] = re.sub(r'\b\w{1,1}\b', '', re.sub('[^A-Za-z0-9]+', ' ', paras[i]))
        
    return paras


# In[31]:


if __name__=='__main__':
    main()

