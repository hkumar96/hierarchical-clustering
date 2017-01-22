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
    '''
    This function calculates cosine similarity between any two vectors(sms)
    Cosine Similarity between any two vector A and B is given by:
            similarity = (A.B)/(norm(A).norm(B))
    The tf_idf matrix(m x n) contains m rows for 'm' unique words and 'n'
    columns for each sms. So, we have 'n' column vectors for 'n' sms'.
    An efficient way to calculate similarities is then using the matrix
    multiplication. If we want to calculate similarity between ith and kth sms
    Let A = tf[:][i] i.e. ith vector and B = tf[:][k] i.e. kth vector.
    Using the above formula we know that similarity matrix will be of size (n x n)
    So, if C has some n column vectors, then dot product of each pair of n vector
    is simply transpose(C).C, where dot(.) represents matrix multiplication.
    The diagonal entries of this matrix are the norm^2 of each of n vector.
    So first we calculate 1/norm for each value and then take its sqaure root.
    After that we once multiply each element at position (i,j) with inv_norm(j) and
    then we take transpose of matrix temp_sim so that each element at position (j,i)
    multiplied by inv_norm(j). Thus calculating the similarity matrix
    '''

    msg_num = len(tf_idf[0])
    tf_idf = np.array(tf_idf)
    temp_sim = np.zeros([msg_num,msg_num])

    # calculating transpose(tf_idf).tf_idf where dot(.) is matrix multiplication
    temp_sim = np.dot(tf_idf.T,tf_idf)
    #norm vector initialized with diagonal entries of temp_sim i.e norm^2
    norm_vec = np.diag(temp_sim)
    #calculated 1/norm^2 for each value
    inv_norm = 1 / norm_vec
    #check if at of the value in inv_norm is infinity, then change it to zero
    inv_norm[np.isinf(inv_norm)] = 0
    #take square root of inv_norm
    inv_norm = np.sqrt(inv_norm)
    #multiply each at element at (i,j) with inv_norm(j)
    temp_sim = temp_sim*inv_norm
    #multiply each element at (j,i) with inv_norm(j)
    temp_sim = temp_sim.T*inv_norm
    return np.array(temp_sim)

def clustering(_sim,threshold):
    '''
    This is the function that handles clustering.
    It uses Johnson's Agglomerative clustering algorithm.
    1. Begin with level(0)=1, where all sms' are in their
    own cluster i.e. one cluster per sms.
    2. Then find the pair of sms that have maximum similarity
    i.e.(i,j) such that sim_temp[i][j] is maximum for all i!=j
    3. Increment the clustering sequence by 1 i.e. m=m+1 and
    update the level with similarity of current clustering
    i.e.  level(m) = sim_temp[i][j].
    4. Update the sim_temp matrix combining ith and jth cluster
    into one cluster.
    5. This is done by updating similarity matrix as follows
     sim_temp[k][j] = max (sim_temp[k][j],sim_temp[k][i])
    6. Then sim_temp[j][k] = sim_temp[k][j] for all k
    7. After this we delete the xth row and column
    8. We repeat this step until maximum similarity is not less
    than the threshold value.
    '''

    labels = [lbl for lbl in range(_sim.shape[1])]

    # Created a temporary similarity matrix to avoid changing original matrix
    sim_temp = np.copy(_sim)

    #initialized all the variables
    m = 0
    level ={}
    max_sim = 0
    level[m] = 1
    clusters = [[lbl] for lbl in labels]

    #loop infinitely until max_sim < threshold
    while True:

        n = sim_temp.shape[1]   #number of sms
        max_sim = 0             #initialized to 0 to find max_sim

    #search for position of maximum similarity (step-2 from above algorithm)
        for i in range(n):
            for j in range(i):
                if max_sim < sim_temp[i,j]:
                    max_sim= sim_temp[i,j]
                    x,y=i,j
    #break if max_sim < threshold
        if max_sim < threshold:
            break
    #increment m by 1 and update level(m) (step-3)
        m = m + 1
        level[m] = max_sim
    #update the similarity matrix (step-4,5)
        for row in sim_temp:
            row[y] = max(row[y],row[x])
    #step 6: copy sim_temp[k][j] to sim_temp[j][k]
        sim_temp[y,:] = sim_temp[:,y]
    #delete the xth row and column (step-7)
        sim_temp=np.delete(sim_temp,x,0)
        sim_temp=np.delete(sim_temp,x,1)
    #update the clusters
        clusters[y].append(clusters.pop(x)[0])
    # print (clusters)
    return clusters
    # print (level)


############### Execution starts from here ################
start_time = time.time()

stopwords =[]
message_list = []
messages = []

#read the stopword list
temp = read_csv("stopword.csv")
for wrd in temp:
    stopwords.append(wrd[0])

temp = read_csv("demo.csv")
for sms in temp:
    messages.append(sms)
    message_list.append(sms[2])

#removing the stopwords
message_list = remove_stopword(message_list,stopwords)
del stopwords[:]

#stemming the list
message_list = stemmer(message_list)

#creating a dictionary from the stemmed words
access = create_corpus(message_list)
corpus = []
num_word = []
for key in access:
    corpus.append(key)
    num_word.append(len(access[key]))

idf = update_idf(message_list,num_word)
tf_idf = calc_tf_idf(message_list,corpus,idf);
sim = calc_similarity(tf_idf)
clusters = clustering(sim,0.7)
file_prefix = "clusters/cluster_"
for i,val in enumerate(clusters):
    filename = file_prefix + str(i) + ".csv"
    with open(filename,'wb') as f:
        writer = csv.writer(f,quotechar=",")
        if not isinstance(val,int):
            for subval in val:
                writer.writerow(messages[subval])
        else:
            writer.writerow(messages[val])

print("--- %s seconds ---" % (time.time() - start_time))
