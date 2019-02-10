
# coding: utf-8

# In[1]:


#Required libraries

import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import itertools
from more_itertools import locate
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec


# In[2]:


def main():
    
    '''
    Function_description: This function is used to read preprocessed train_paras.txt and tags.pkl.
    It is also used to build and save FastText and Word2Vec models on paras.
    
    Input_to_function: None
    
    Return_values_from_function: None
        
    dependencies: calls tags_preprocessing()
    
    '''
    
    #Read pre-processed train_paras.txt
    with open('train_paras.txt','r') as f:
        paras=f.readlines()

    for i in range(0,len(paras)):
        paras[i]=paras[i][2:].split("', '")
        paras[i][-1]=paras[i][-1].replace("']\n",'')
    
    #Read pre-processed tags.pkl
    with open("tags.pkl", "rb") as fp:
        tags = pickle.load(fp)
    
    #Set it to True to build a fasttext model
    if False:
        fast_text_model(paras)
    
    #Set it to True to build a Word2Vec model
    if False:
        model_wv=word2vec_model(paras)
    
    #Load already built Word2vec model
    model_wv = Word2Vec.load('word2vec.model')
    
    #Call tags_preprocessing()
    new_tags=tags_preprocessing(tags)
    
    #Call new_target()
    new_target(new_tags)


# In[3]:


def fast_text_model(paras):
    #Building Fasttext model and save
    model_ft = FastText(paras, size=150, window=5, min_count=3)
    model_ft.save('fasttext.model')


# In[ ]:


def word2vec_model(paras):
    #Building Word2vec model and save
    model_wv=Word2Vec(paras,size=150,window=6,min_count=3,iter=40)
    model_wv.save('word2vec.model')
    return model_wv


# In[5]:


def tags_preprocessing(tags):
    
    '''
    Function_description: This function is used to create new tags. If tags are '-' separated
    they are considered as new type of tags and classified to another class.
    
    Ex: if tag is python - word in para with python is tagged as class 1
        if tag is bit-manipulation - word in para after splitting is tagged as bit - class 2
        and manipulation - class 2
        if tag is google-maps-api - word in para after splitting is tagged as google - class 3,
        maps is tagged - class 3 and api is tagged - class 3
                                          
    
    Input_to_function: tags
    
    Return_values_from_function: new tags
        
    dependencies: None
    
    '''
    new_tags=[]
    #creating special new tags for '-' separated tags. Ex: bit-manipulation is 
    #transformed to $bit, $manipulation.
    
    for i in range(0,len(tags)):
        tag=[]
        for j in range(0,len(tags[i])):
            sp=tags[i][j].split('-')
            
            #if two words are joined by '-' in tag then '$' is added to each sub tag
            if len(sp)==2:
                tag.append('$'+sp[0])
                tag.append('$'+sp[1])
            
            #if three words are joined by '-' in tag then '#' is added to each sub tag
            elif len(sp)==3:
                tag.append('#'+sp[0])
                tag.append('#'+sp[1])
                tag.append('#'+sp[2])
            else:
                tag.append(sp[0])
        
        new_tags.append(tag)
    
    #return new tags
    return new_tags


# In[7]:


def tagger(value,word):
    temp = []
    #Search for 10 most similar word2vec words
    try:
        values = [i[0] for i in model_wv.wv.most_similar(word)]
    except KeyError:
        return []
    
    #Find if tag
    for i in range(10):
        temp.append(list(locate(paras[value], lambda x: x == values[i])))
    
    return list(itertools.chain.from_iterable(temp))


# In[ ]:


def new_target(new_tags):
    
    '''
    Function_description: This function creates numeric encoding of tags for each
    word in para.
    
    Ex: I write code in Python on my LINUX machine.
    If tags of above sentence are Python and LINUX
    It is numerically encoded as : 0 0 0 0 1 0 0 1 0
                                          
    Ex: I used google maps api to build this android project.
    If tags of above sentence are google-maps-api and android
    It is numerically encoded as : 0 0 4 4 4 0 0 0 1 0    
    Input_to_function: tags
    
    Return_values_from_function: new tags
        
    dependencies: one_hot_target_create()
    
    '''
    
    
    target=[]
    
    #Creating numeric target for each tag.
    for i in range(0,len(new_tags)):
        
        #Creating a target for each word in the para
        target_temp = np.array([0] * len(paras[i]))
        
        #For each tag in all the tags corresponding to a para
        for j in range(0,len(new_tags[i])):
            
            #if tag contains '$' and any word in para matches with top 10 similar words
            #of tag then target is codes as 3
            if new_tags[i][j][0]=='$':
                temp=list(locate(paras[i], lambda x: x == new_tags[i][j][1:]))
                target_temp[temp]=3
            
            #if tag contains '#' and any word in para matches with top 10 similar words
            #of tag then target is codes as 4
            elif new_tags[i][j][0]=='#':
                temp=list(locate(paras[i], lambda x: x == new_tags[i][j][1:]))
                target_temp[temp]=4
            
            #if tag doesn't contain any special symbol
            else:
                
                #search for any word in para matching with top 10 similar words of tag
                #then target is codes as 2
                temp=list(locate(paras[i], lambda x: x == new_tags[i][j]))
                
                #If word in para is from any of top 10 similar words of tag
                if len(temp)==0:
                    temp = tagger(i,new_tags[i][j])
                    target_temp[temp] = 2
                
                #If word in para is exactly the tag itself
                else:
                    target_temp[temp]=1
        
        target.append(target_temp)
    
    #New target needs to be one hot encoded for building the model.
    one_hot_target_create(target)


# In[12]:


def one_hot_target_create(target)

    '''
    Function_description: This function creates one hot encoding for each
    word in the para and saves it in the file for furthur model building.
    
    For each word in para a 5-dim target vector is created.
    0-no tag
    1-if exact word is tag
    2-if similar word is tag
    3-if word is part of '-' separated tag(len=2)
    4-if word is part of '-' separated tag(len=3)
    
    Input_to_function: numerically encodes target
    
    Return_values_from_function: None
        
    dependencies: None
    
    '''
    final_target = []

    #For each sequence in the target
    for i in range(len(target)):
        
        #For each tag corresponding to each word in para
        feed_target = np.zeros([len(target[i]),5])
        
        #One hot encoding
        for j in range(len(target[i])):
            if target[i][j] == 0:
                feed_target[j][0] = 1
            if target[i][j] == 1:
                feed_target[j][1] = 1
            if target[i][j] == 2:
                feed_target[j][2] = 1
            if target[i][j] == 3:
                feed_target[j][3] = 1
            if target[i][j] == 4:
                feed_target[j][4] = 1
        final_target.append(feed_target)
    
    #Writing one hot target to file for furthur model building
    with open("one_hot_target.txt", "w") as f:
        for s in final_target:
            for k in s:
                f.write(str(k) +"\n")

