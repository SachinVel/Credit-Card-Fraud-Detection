#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('C:/Users/sachin/Desktop/Data analytics/clean2.xls')
df.head()


# In[3]:


trainRow = df.iloc[1:20000,:]
testRow = df.iloc[20001:30000,:]
x_train = trainRow[["LIMIT_BAL","Pay_mean","bill_mean"]]
y_train = trainRow[["default.payment.next.month"]]
x_test = testRow[["LIMIT_BAL","Pay_mean","bill_mean"]]
y_test = testRow[["default.payment.next.month"]]


# In[4]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)


# In[5]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred)*100)


# In[8]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)
tree.fit(x_train, y_train) 
y_pred = tree.predict(x_test)


# In[9]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred)*100)


# In[ ]:




