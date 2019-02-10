
# coding: utf-8

# In[1]:


#Required libraries

import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import itertools
import keras
from sklearn.model_selection import train_test_split
from gensim.models import FastText
from copy import deepcopy
import keras_contrib
from sklearn.metrics import f1_score,accuracy_score
from keras.models import load_model
from gensim.models import Word2Vec
from more_itertools import locate


# In[2]:


def main():
    
    #Read preprocessed tags
    with open("tags.pkl", "rb") as fp:
        tags = pickle.load(fp)
    
    #Read preprocessed train_paras.txt
    with open('train_paras.txt','r') as f:
        data=f.readlines()

    for i in range(0,len(data)):
        data[i]=data[i][2:].split("', '")
        data[i][-1]=data[i][-1].replace("']\n",'')
    
    #Read one hot target
    with open('one_hot_target.txt','r') as f:
        full_target=f.readlines()

    target=[]
    i=0
    k=0
    while(i<len(full_target)):
        new_sen=[]
        for j in range(0,len(data[k])):
            x=full_target[i+j].replace("[",'')
            x=x.replace("]\n",'')
            x=x.split(' ')
            x=[float(num) for num in x if num not in ""]
            new_sen.append(x)
        i+=len(data[k])
        k+=1
        target.append(np.array(new_sen))
    
    #Convert target to list
    for i in range(0,len(target)):
        target[i]=target[i].tolist()
    
    #Load fasttext and word2vec models
    fasttext_model = FastText.load('fasttext.model')
    w2v_model=Word2Vec.load('word2vec.model') 
    
    #Padding both data and target to fixed size of 300
    data,target=padding(data,target)
    
    #Calling train validation split
    x_train,y_train,x_test,y_test,tags_test=train_test_split(data,target)
    
    #Building model architecture
    model=model_define()
    
    #Fitting the model for training data
    model.fit_generator(generator=generator(64,x_train,y_train),epochs=1,steps_per_epoch=len(data)/64)
    
    #Saving model weights
    model.save_weights('keras_full_model_weights.h5')
    
    #Sorting tags based on frequency
    set_of_all_tags=sorted_tags(tags)
    
    #Predicting tags on test data
    predicted_tags=final_pred(x_test,model,set_of_all_tags,w2v_model)
    
    #Combining predicted tags with tags predicted from code_classification
    predicted_tags=code_classify(x_test,predicted_tags)
    
    #Computing f1 score
    avg_f1=0
    for i in range(0,len(x_test)):
        avg_f1+=f1score(predicted_tags[i],tags_test[i])
    print(avg_f1/len(x_test))


# In[3]:


def padding(data,target):
    
    #iterating over each sentence
    for i in range(0,len(data)):
        
        #slicing to 300 size para
        data[i]=data[i][0:300]
        leng=300-len(data[i])
        #padding
        data[i]=data[i]+leng*['pad']
        #slicing target to 300 size
        target[i]=target[i][0:300]
        target[i]+=([[1.,  0.,  0.,  0.,  0.]]*leng)
    
    #returning 300 sized paragraph and target
    return data,target


# In[ ]:


def train_test_split(data,target):
    
    #80% of data for training
    x_train=data[0:int(0.80*len(data))]
    y_train=target[0:int(0.80*len(data))]
    
    #20% of data for validation
    x_test=data[len(x_train):]
    y_test=target[len(x_train):]

    tags_test=tags[len(x_train):]
    
    return x_train,y_train,x_test,y_test,tags_test


# In[ ]:


def model_define():
    
    #Architecture of the model
    input_text = keras.layers.Input(shape=(300,300))
    lstm_1 = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))(input_text)
    drop = keras.layers.Dropout(0.2)(lstm_1)
    dense = keras.layers.TimeDistributed(keras.layers.Dense(50, activation="relu"))(drop)
    out=keras.layers.TimeDistributed(keras.layers.Dense(5,activation='softmax'))(dense)
    
    #Defining the model
    model = keras.Model(inputs = input_text,outputs=out)
    
    #Compiling the model
    model.compile(optimizer="adam", loss='categorical_crossentropy')
    
    return model
    #model.load_weights('keras_full_model_weights.h5')


# In[ ]:


def generator(batch_size,from_list_x,from_list_y):
    
    #Generator function model training
    assert len(from_list_x) == len(from_list_y)
    total_size = len(from_list_x)

    while True:

        for i in range(0,total_size,batch_size):
            yield np.array(transformation(deepcopy(from_list_x[i:i+batch_size]))),np.array(from_list_y[i:i+batch_size])


# In[4]:


def transformation(x):
    
    #Extracting word2vec and fasttext vectors for each word in para
    for i in range(len(x)):
        for j in range(300):
            try:
                #word2vec feature
                w2v = w2v_model.wv[x[i][j]]
            except:
                w2v = np.zeros([150])
            try:
                #fasttext feature
                ft = fasttext_model.wv[x[i][j]]
            except:
                ft = np.zeros([150])
            
            #Concatenating word2vec and fasttext features
            x[i][j] = np.concatenate([w2v,ft])
            
    return np.array(x)


# In[ ]:


def sorted_tags(tags):
    merged = list(itertools.chain.from_iterable(tags))

    dictionary_tags={}
    for i in range(0,len(merged)):
        try:
            dictionary_tags[merged[i]]+=1
        except:
            dictionary_tags[merged[i]]=1
        
    sorted_by_value = sorted(dictionary_tags.items(), key=lambda kv: kv[1],reverse=True)

    set_of_all_tags=[]
    for i in range(0,len(sorted_by_value)):
        if sorted_by_value[i][0] not in set_of_all_tags:
            set_of_all_tags.append(sorted_by_value[i][0])
            
    return set_of_all_tags


# In[ ]:


def reverse_lookup(word,set_of_all_tags,w2v_model):
    try:
        values = [i[0] for i in w2v_model.wv.most_similar(word)]
    except KeyError:
        return 0
    
    for i in range(len(values)):
        try:
            lookup = set_of_all_tags.index(values[i])
            return set_of_all_tags[lookup]
        except ValueError:
            pass


# In[ ]:


def look_up_two(word,set_of_all_tags):
    for i in range(0,len(set_of_all_tags)):
        x=set_of_all_tags[i].split('-')
        if word in x and len(x)==2:
            return [set_of_all_tags[i]]
    return []


# In[ ]:


def look_up_three(word,set_of_all_tags):
    for i in range(0,len(set_of_all_tags)):
        x=set_of_all_tags[i].split('-')
        if word in x and len(x)==3:
            return [set_of_all_tags[i]]
    return []


# In[ ]:


def predict(pred_data,model,set_of_all_tags,w2v_model):
    
    dat = transformation(deepcopy(pred_data))
    result = model.predict(dat)
    
    new_res=[]
    for i in range(0,len(result)):
        pred_data[i]=np.array(pred_data[i])
        res=[]
        for j in range(0,len(result[i])):
            res.append(np.argmax(result[i][j]))
        new_res.append(res) 
        
    del result
    
    main_tags = []
    sub_tags = []
    
    for i in range(0,len(new_res)):
        main_tags.append(pred_data[i][np.where(np.array(new_res[i])==1)[0]])
        sub_tags.append(pred_data[i][np.where(np.array(new_res[i])==2)[0]])
        
    sub_tags_all=[]
    for i in range(0,len(sub_tags)):
        sub_tags_temp = []
        j=0
        while(j<len(sub_tags[i])):
            z = reverse_lookup(sub_tags[i][j],set_of_all_tags,w2v_model)
            if z != 0:
                sub_tags_temp.append(z)
            j+=1
        sub_tags_all.append(sub_tags_temp)
    
    
    two_tags_all=[]
    for i in range(0,len(new_res)):
        two_tags_indices=np.where(np.array(new_res[i])==3)[0]
        two_tags_temp=[]
        for j in range(0,len(two_tags_indices)-1):
            doub=look_up_two(pred_data[i][two_tags_indices[j]],set_of_all_tags)
            try:
                dou=doub[0].split('-')
                if dou[1]==pred_data[i][two_tags_indices[j+1]]:
                    two_tags_temp+=doub
            except:
                pass
        two_tags_all.append(two_tags_temp)
            
        
    three_tags_all=[]
    for i in range(0,len(new_res)):
        three_tags_indices=np.where(np.array(new_res[i])==4)[0]
        three_tags_temp=[]
        for j in range(0,len(three_tags_indices)-1):
            thre=look_up_three(pred_data[i][three_tags_indices[j]],set_of_all_tags)
            try:
                three=thre[0].split('-')
                if three[1]==pred_data[i][three_tags_indices[j+1]] or three[2]==pred_data[i][three_tags_indices[j+2]]:
                    three_tags_temp+=thre
            except:
                pass
        three_tags_all.append(three_tags_temp)
    
    full_tags=[]
    for i in range(0,len(pred_data)):        
        full = list(main_tags[i]) + sub_tags_all[i]+two_tags_all[i]+three_tags_all[i]
        full_tags.append(list(set(full)))
    
    
    return full_tags


# In[ ]:


def final_pred(data,model,set_of_all_tags,w2v_model):
    i=0
    predicted_tags=[]
    while(i<len(data)):
        pred=predict(data[i:i+64],model,set_of_all_tags,w2v_model)
        for j in range(0,len(pred)):
            if 'angular' in pred[j]:
                ind=pred[j].index('angular')
                pred[ind]=['angularjs']
            if 'ruby' in pred[j]:
                ind=pred[j].index('ruby')
                pred[ind]=['ruby-on-rails']
        i+=64
        predicted_tags+=pred
    return predicted_tags


# In[ ]:


def code_classify_model(data,predicted_tags):
    for i in range(0,len(data)):
        data[i]=" ".join(word for word in data[i])

    with open("vectorizer_1.pkl", "rb") as fp:
        vectorizer_1 = pickle.load(fp)
    with open("vectorizer_2.pkl", "rb") as fp:
        vectorizer_2 = pickle.load(fp)
    with open("model_codes_1.pkl", "rb") as fp:
        model_1 = pickle.load(fp)
    with open("model_codes_2.pkl", "rb") as fp:
        model_2 = pickle.load(fp)

    for i in range(0,len(data)):
        prob1=model_1.predict_proba(vectorizer_1.transform([data[i]]))
        prob2=model_2.predict_proba(vectorizer_2.transform([data[i]]))
        ind1=np.argmax(prob1)
        ind2=np.argmax(prob2)
        if prob1[0][ind1]>0.40:
            predicted_tags[i].append((model_1.predict(vectorizer_1.transform([data[i]]))[0]))
        elif prob2[0][ind2]>0.60:
            predicted_tags[i].append((model_2.predict(vectorizer_2.transform([data[i]]))[0]))
    return predicted_tags


# In[ ]:


if __name__=="__main__":
    main()

