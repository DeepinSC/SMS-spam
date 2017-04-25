# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 08:53:33 2017

@author: Jason
"""
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

def shuffle(X, y):
    n = len(y)
    index = np.arange(0,n,1)
    random.shuffle(index)
    
    x_return=y_return=[]
    for i in range(n):
        print(i,index[i])
        x_return.append(X[index[i]])
        y_return.append(y[index[i]])
    
    return x_return,y_return

#数据处理
def get_data():
    # csv文件预处理
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    data = data.rename(columns = {'v1':'label','v2':'text'})
    data['length'] = data['text'].apply(len)
    data['label_num'] = data.label.map({'ham':0, 'spam':1})
    print('短信分类统计：')
    print(data.label.value_counts())
    
    spam = data[data.label_num == 1]["text"]
    ham = data[data.label_num == 0]["text"]
    m = spam.count()
    n = ham.count()
    
    X_test = spam[:int(m*0.3+1)].append(ham[:int(n*0.3+1)])
    y_test = [1]* (int(m*0.3+1))
    y_test.extend([0]*(int(n*0.3+1)))
    X_train = spam[int(m*0.3+1):].append(ham[int(n*0.3+1):])
    y_train = [1]*(m-int(m*0.3+1))
    y_train.extend([0]*(n-int(n*0.3+1)))
    
    X_train,y_train= shuffle(list(X_train), y_train) 
    X_test,y_test= shuffle(list(X_test), y_test)
    # 划分训练集、测试集
    X_train,X_test,y_train,y_test = train_test_split(
            data["text"],data["label_num"], test_size = 0.3, random_state = 10)
    
    # 训练文本向量化工具
    vect = CountVectorizer(stop_words='english')
    vect.fit(X_train)
    
    # 利用训练好的工具将文本向量化为稀疏矩阵
    X_train_df = vect.transform(X_train)
    X_test_df = vect.transform(X_test)
    
    return X_train_df,X_test_df,list(y_train),list(y_test)




#查准率，召回率以及F1
def score(y,y_pred):
    ss = sn = 0#垃圾邮件被分类为垃圾邮件数目，垃圾邮件被分类为正常邮件数目
    nn = ns = 0#正常邮件被分类为正常邮件数目，正常邮件被分类为垃圾邮件数目
    for i in range(len(y)):
        if int(y[i])==1 and int(y_pred[i])==1:
            ss+=1
        elif int(y[i])==1 and int(y_pred[i])==0:
            sn+=1
        elif int(y[i])==0 and int(y_pred[i])==0:
            nn+=1
        else:
            ns+=1
    precision=ss/(ss+ns)
    recall=ss/(ss+sn)
    print(ss,sn,nn,ns)
    return 2*precision*recall/(precision+recall)
#        return ss, nn, (ss+nn)/(ss+nn+sn+ns)

X_train,X_test,y_train,y_test = get_data()

# 定义分类器
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier()
lrc = LogisticRegression()
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
mlp = MLPClassifier()

clfs = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 
        'LR': lrc, 'RF': rfc, 'AdaBoost': abc, 'MLP':mlp}

# 训练、测试分类器
pred_scores = []
for k,v in clfs.items():
    v.fit(X_train, y_train)
    pred = v.predict(X_test)
    pred_scores.append((k, [score(y_test,pred)]))
    
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])


df.plot(kind='bar', ylim=(0.8,1.0), figsize=(11,6), align='center', colormap="Accent")
plt.xticks(np.arange(9), df.index)
plt.title('Distribution by Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)