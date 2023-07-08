#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import KFold
#importing all required libraries and their functions

c = 0
d = {}

arr_y =[]
for i in range(1,24):
    for j in range(15):   
        arr_y.append(i)

y= np.array(arr_y)

arr_y =[]
for i in range(1,24):
    for j in range(5):   #creating a list of size equal to testing data in such a way so that i can check accuracy of later
        arr_y.append(i)

y_new = np.array(arr_y) #making the list a numpy array

author = ['./articles/train/author_1', './articles/train/author_2', './articles/train/author_3',
          './articles/train/author_4', './articles/train/author_5', './articles/train/author_6', './articles/train/author_7',
              './articles/train/author_8', './articles/train/author_9', './articles/train/author_10',
              './articles/train/author_11', './articles/train/author_12',
              './articles/train/author_13', './articles/train/author_14',
              './articles/train/author_15', './articles/train/author_16',
              './articles/train/author_17', './articles/train/author_18',
              './articles/train/author_19', './articles/train/author_20',
              './articles/train/author_21', './articles/train/author_22', './articles/train/author_23']

arr = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt', '10.txt', '11.txt', '12.txt', '13.txt', '14.txt', '15.txt']


collection = []

for a in range(23):
    feat_vect = []
    for m in range(15):        
        filename = author[a] + '/' + arr[m]            
        with open(filename, encoding="utf8") as f:
            collection.append(f.read())
       #loop to read articles and add article to collection      
       
       

vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (1,2), max_features = None)
X = vectorizer.fit_transform(collection)
feat_names = vectorizer.get_feature_names()
#created a tfidf vector of collection


sum2=0
for i in range(15):
    sum1=0
    X_arr = X.toarray()
    af = KFold(n_splits=15, shuffle = True)
    #af = KFold(n_splits=5, shuffle = True)
    af.get_n_splits(X_arr)
    c=c+1
    print("TEST {} :".format(c))
    for train_index, test_index in af.split(X_arr):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #kf = LeaveOneOut()
    #for train_index, test_index in kf.split(X_arr):
        # X_train, X_test = X_arr[train_index], X_arr[test_index]
        #y_train, y_test = y[train_index], y[test_index]
    
#LEAVE ONE OUT TAKES LOT OF TIME 
#DONT RUN LEAVE ONE OUT 15 TIMES TO GET AVERAGE  AS IT WILL TAKE LOT OF TIME AND RESULT IS MOSTLY CONSTANT

        C = 100
        clf = svm.SVC(C=C,kernel='linear')
        clf.fit(X_train, y_train)
        pr = clf.predict(X_test)
        sum1 = sum1 + (accuracy_score(pr,y_test))
        
    sum2=sum2 + sum1/15
    #sum2=sum2 + sum1/5
    print(sum1/15)
    #print(sum1/5)
    #print(sum1/345)
    
print("AVERAGE:")    
print(sum2/15)


C = 100
clf = svm.SVC(C=C,kernel='linear')
clf.fit(X_arr, y)


########## TESTING SECTION #########


author = ['./articles/test/author_1', './articles/test/author_2', './articles/test/author_3',
          './articles/test/author_4', './articles/test/author_5', './articles/test/author_6',
          './articles/test/author_7',
              './articles/test/author_8', './articles/test/author_9', './articles/test/author_10',
              './articles/test/author_11', './articles/test/author_12',
              './articles/test/author_13', './articles/test/author_14',
              './articles/test/author_15', './articles/test/author_16',
              './articles/test/author_17', './articles/test/author_18',
              './articles/test/author_19', './articles/test/author_20',
              './articles/test/author_21', './articles/test/author_22', './articles/test/author_23']

arr = ['16.txt', '17.txt', '18.txt', '19.txt', '20.txt']


collection = []

for a in range(23):
    feat_vect = []
    for m in range(5):        
        filename = author[a] + '/' + arr[m]            
        with open(filename, encoding="utf8") as f:          
            collection.append(f.read())
         #loop to read articles and add article to collection  
       

vectorizer = TfidfVectorizer(vocabulary = feat_names, stop_words = 'english', ngram_range = (1,2), max_features = None)
X = vectorizer.fit_transform(collection)
feat_names = vectorizer.get_feature_names()
#created a tfidf vector of collection
X_arr_1 = X.toarray()

predictions = clf.predict(X_arr_1)
print(predictions)
print(y_new)
print(accuracy_score(predictions,y_new))


# In[ ]:




