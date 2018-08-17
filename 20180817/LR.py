
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import gc


# In[2]:

train_data = open('train_set.csv')


# In[3]:

lines = train_data.readlines()
X_train = []
y_train = []
cnt = 0
for line in lines:
    cnt += 1
    if cnt==1:
        continue
    tmp = line.split(',')
    X_train.append(tmp[2])
    y_train.append(int(tmp[3])-1)
train_data.close()
del train_data,lines,cnt
gc.collect()


# In[12]:

vectorizer = CountVectorizer(ngram_range=(1,2),min_df=3,max_df=0.9,max_features=100000)
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)


# In[18]:

lg = LogisticRegression(C=4,dual=True)
lg.fit(X_train,y_train)


# In[19]:

del X_train,y_train
gc.collect()


# In[21]:

test_data = open('test_set.csv')
lines = test_data.readlines()
X_test = []
id = []
cnt = 0
for line in lines:
    cnt += 1
    if cnt==1:
        continue
    tmp = line.split(',')
    X_test.append(tmp[2])
    id.append(tmp[0])
test_data.close()
del test_data,lines,cnt
gc.collect()


# In[24]:

X_test = vectorizer.transform(X_test)


# In[25]:

y_test = lg.predict(X_test) + 1


# In[26]:

ans = pd.DataFrame({'id':id,'class':y_test})
output = ans[['id','class']]
output.to_csv("result.csv",index = False)

