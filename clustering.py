from __future__ import print_function
import csv
from nltk.stem.porter import *
import numpy as np

def read_csv(filename="sms.csv"):
    with open(filename,'rb') as csvfile:
        msg_reader = csv.reader(csvfile, quotechar='"')
        return [row for row in msg_reader]

def remove_stopword(msg_list,stopwords):
    temp = []
    for msg in msg_list:
        msg = [word for word in msg if word.lower() not in stopwords]
        temp.append(msg)
    return temp

def stem(msg_list):
    stemmer = PorterStemmer()
    temp = []
    for msg in msg_list:
        msg = [stemmer.stem(word) for word in msg]
        temp.append(msg)
    return temp

def create_corpus(msg_list):
    temp_cp = []
    for msg in msg_list:
        for word in msg:
            if word not in temp_cp:
                temp_cp.append(word)
    return temp_cp

def update_tf(msg_list,corpus):
    temp_tf = np.zeros(shape=(len(corpus),len(msg_list)))
    for i in range(len(corpus)):
        for j in range(len(msg_list)):
            temp_tf[i][j] = msg_list[j].count(corpus[i])*1.0/len(msg_list[j])
    return np.array(temp_tf)

def update_idf(msg_list,corpus):
    temp_idf = np.zeros(len(corpus))
    for i in range(len(corpus)):
        matches = len([True for msg in msg_list if corpus[i] in msg])
        temp_idf[i] = (np.log10(len(msg_list)*1.0 / (1 + matches)))
    return np.array(temp_idf)

def calc_tf_idf(tf,idf):
    temp_tf_idf = np.zeros(tf.shape)
    for i in range(tf.shape[0]):
        for j in range(tf.shape[1]):
            temp_tf_idf[i][j] = tf[i][j]*idf[i]
    return np.array(temp_tf_idf)

def calc_similarity(tf_idf):
    msg_num = tf_idf.shape[1]
    temp_sim = np.zeros([msg_num,msg_num])
    norm_vec = np.linalg.norm(tf_idf,axis=0)
    for i in range(msg_num):
        for k in range(msg_num):
            temp_sim[i,k] = 1.0*np.sum(np.dot(tf_idf[:,i],tf_idf[:,k]))/(norm_vec[i]*norm_vec[k])
    return np.array(temp_sim)

stopwords =[]
message_list = []

temp = read_csv("stopword.csv")
for wrd in temp:
    stopwords.append(wrd[0])
# print (stopwords)
temp = read_csv("demo.csv")
for sms in temp:
    message_list.append(sms[2].split()) #tokenization not done properly

message_list = remove_stopword(message_list,stopwords)
message_list = stem(message_list)

corpus = create_corpus(message_list)
tf = update_tf(message_list,corpus)
idf = update_idf(message_list,corpus)
tf_idf = calc_tf_idf(tf,idf);
sim = calc_similarity(tf_idf)

# print (corpus)
# print (idf)
print (tf_idf.shape)
print(sim)
