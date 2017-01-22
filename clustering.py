from __future__ import print_function
import csv
import re
from stemming.porter2 import stem
import numpy as np
import time

def read_csv(filename="sms.csv"):
    with open(filename,'rb') as csvfile:
        msg_reader = csv.reader(csvfile, quotechar='"')
        return [row for row in msg_reader]

def remove_stopword(msg_list,stopwords):
    temp = []
    for msg in msg_list:
        msg = " ".join([word for word in re.split(';|\ |,|\?|@|\*|$|\"|\n',msg) if word.lower() not in stopwords])
        temp.append(msg)
    return temp

def stemmer(msg_list):
    temp = []
    for msg in msg_list:
        msg = [stem(word) for word in msg.split()]
        temp.append(msg)
    return temp

def create_corpus(msg_list):
    temp_cp = {}
    for i,msg in enumerate(msg_list):
        for word in msg:
            if word not in temp_cp:
                temp_cp[word] = {i}
            else:
                temp_cp[word].add(i)
    return temp_cp

def ret_tf(msg_list,corpus,x,y):
    if (len(msg_list[y]) == 0):
        return 0
    return (1.0*msg_list[y].count(corpus[x]) / len(msg_list[y]))

def update_idf(msg_list,num_word):
    temp_idf = np.zeros(len(num_word))
    for i in range(len(num_word)):
        temp_idf[i] = (np.log10(len(msg_list)*1.0 / (1 + num_word[i])))
    return np.array(temp_idf)

def calc_tf_idf(msg_list,corpus,idf):
    temp_tf_idf = []
    for i in range(len(corpus)):
        temp_tf_idf.append([ret_tf(msg_list,corpus,i,j)*idf[i] for j in range(len(msg_list))])
    return temp_tf_idf

def calc_similarity(tf_idf):
    msg_num = len(tf_idf[0])
    tf_idf = np.array(tf_idf)
    temp_sim = np.zeros([msg_num,msg_num])
    temp_sim = np.dot(tf_idf.T,tf_idf)
    norm_vec = np.diag(temp_sim)
    inv_norm = 1 / norm_vec
    inv_norm[np.isinf(inv_norm)] = 0
    inv_norm = np.sqrt(inv_norm)
    temp_sim = temp_sim*inv_norm
    temp_sim = temp_sim.T*inv_norm
    return np.array(temp_sim)

def clustering(_sim,threshold):
    labels = [lbl for lbl in range(_sim.shape[1])]
    sim_temp = np.copy(_sim)
    m=0
    level ={}
    max_sim = 0
    level[m] = 1
    clusters = labels
    while True:
        n = sim_temp.shape[1]
        max_sim = 0
        for i in range(n):
            for j in range(i):
                if max_sim < sim_temp[i,j]:
                    max_sim= sim_temp[i,j]
                    x,y=i,j
        if max_sim < threshold:
            break
        m = m + 1
        level[m] = max_sim
        for row in sim_temp:
            row[y] = max(row[y],row[x])
        sim_temp[y,:] = sim_temp[:,y]
        sim_temp=np.delete(sim_temp,x,0)
        sim_temp=np.delete(sim_temp,x,1)
        clusters[y] = (clusters[y],clusters.pop(x))
    print (clusters)
    # print (level)



start_time = time.time()
stopwords =[]
message_list = []

temp = read_csv("stopword.csv")
for wrd in temp:
    stopwords.append(wrd[0])

temp = read_csv("demo.csv")
for sms in temp:
    message_list.append(sms[2])

message_list = remove_stopword(message_list,stopwords)
message_list = stemmer(message_list)


access = create_corpus(message_list)
corpus = []
num_word = []
for key in access:
    corpus.append(key)
    num_word.append(len(access[key]))

idf = update_idf(message_list,num_word)
tf_idf = calc_tf_idf(message_list,corpus,idf);
sim = calc_similarity(tf_idf)
clustering(sim,0.7)

print("--- %s seconds ---" % (time.time() - start_time))
