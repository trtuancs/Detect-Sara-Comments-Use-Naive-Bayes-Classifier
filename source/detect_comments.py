from __future__ import print_function
import nltk
import re
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score

train_data_normal = 'training_normal_comments.txt'
train_data_sara = 'training_sara_comments.txt'
test_data_normal = 'test_normal_comments.txt'
test_data_sara = 'test_sara_comments.txt'
stop_words = 'stop_words.txt'
id_full ='id_full.txt'
train_features ='train_features.txt'
test_features = 'test_features.txt'

def create_label(data_normal,data_sara):
    num_lines_normal = 0
    num_lines_sara = 0
    
    with open(data_normal, 'r',encoding="utf8") as f:
        for line in f:
            num_lines_normal += 1
    with open(data_sara, 'r',encoding="utf8") as f:
        for line in f:
            num_lines_sara += 1
    label =[]
    for j in range(num_lines_normal):
        label.append(0)
    for i in range(num_lines_normal,num_lines_normal+num_lines_sara):
        label.append(1)
    return label

train_label = create_label(train_data_normal,train_data_sara)
test_label = create_label(test_data_normal,test_data_sara)

def create_list_stop_words(stop_words):
    list_stop_words=[]
    with open(stop_words, 'r',encoding="utf8") as f:
        for line in f:
            list_stop_words.append(line.strip())
    return list_stop_words

Stop_Words = create_list_stop_words(stop_words)

def create_dic(dic_):
    list_dic = []
    with open (dic_,'r',encoding= "utf8") as f:
        for line in f:
            string_line = ''.join([i for i in line if not i.isdigit()])
            list_dic.append(string_line.strip())
    return list_dic

Dic_ = create_dic(id_full)  # Dictionary
nwords =len(Dic_)

def process_data(data_normal,data_sara,name):
    num_line = 0    # Line number
    file_data = open( name + "_features.txt",'w')
    with open(data_normal, 'r',encoding="utf8") as f:
        for line in f:
            num_line = num_line +1
            output_1 = re.sub('[^A-Za-z]+', ' ', line)  # Loc ky tu dac biet va so
            output_2 = ' '.join([word for word in output_1.lower().split() if word not in Stop_Words])  # Loc Stop_words
            output_3 = nltk.word_tokenize(output_2) # Tach tu
            output_3.sort() # Sap xep tu
            output_4 = dict(Counter(output_3))  # Tao dictionary voi value la freq
            for word in output_4:
                if word in Dic_:
                    # file_data.write(str(num_line) + " " + str(Dic_.index(word)+1) + " " + str(output_4[word]) + "\n")
                    file_data.write(str(num_line) + " ")
                    file_data.write(str(Dic_.index(word)+1) + " ")
                    file_data.write(str(output_4[word]) + "\n")
    with open(data_sara, 'r',encoding="utf8") as f:
        for line in f:
            num_line = num_line +1
            output_1 = re.sub('[^A-Za-z]+', ' ', line)  # Loc ky tu dac biet va so
            output_2 = ' '.join([word for word in output_1.lower().split() if word not in Stop_Words])  # Loc Stop_words
            output_3 = nltk.word_tokenize(output_2)
            output_3.sort()
            output_4 = dict(Counter(output_3))
            for word in output_4:
                if word in Dic_:
                    file_data.write(str(num_line) + " " + str(Dic_.index(word)+1) + " " + str(output_4[word]) + "\n")
    file_data.close()


process_data(train_data_normal,train_data_sara,"train")
process_data(test_data_normal,test_data_sara,"test")

def read_data(data_features,name):
    with open(data_features) as f:
        content = f.readlines()
    # remove ’\n’ at the end of each line
    content = [x.strip() for x in content]
    dat = np.zeros((len(content), 3), dtype = int)
    for i, line in enumerate(content):
        a = line.split(' ')
        dat[i, :] = np.array([int(a[0]), int(a[1]), int(a[2])])
    # remember to -1 at coordinate since we’re in Python
    data = coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)),\
            shape=(len(name), nwords))
    return (data)

train_data = read_data(train_features,train_label)
test_data = read_data(test_features,test_label)

clf = MultinomialNB()
clf.fit(train_data,train_label)
y_pred = clf.predict(test_data)

print('Training size = %d, accuracy = %.2f%%' % \
    (train_data.shape[0],accuracy_score(test_label, y_pred)*100))