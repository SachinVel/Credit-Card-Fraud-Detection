#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('C:/Users/sachin/Desktop/Data analytics/clean2.xls')
df.head()
df1 = df[["Pay_mean","bill_mean"]]


# In[3]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,max_iter=500,algorithm='auto')
kmeans.fit(df1)


# In[4]:


x = kmeans.fit_predict(df1)
df1['cluster']=x
df1.head()


# In[5]:



from matplotlib import pyplot as plt

plt.style.use('ggplot')
f1 = df1['Pay_mean'].values
f2 = df1['bill_mean'].values
plt.scatter(f1, f2, c=df1['cluster'], s=7)


# In[10]:


from sklearn.cluster import AgglomerativeClustering
Hclustering = AgglomerativeClustering(n_clusters=10,affinity='euclidean', linkage='ward')
Hclustering.fit(df1)
df1['hcluster'] = Hclustering.labels_


# In[11]:


from matplotlib import pyplot as plt

plt.style.use('ggplot')
f1 = df1['Pay_mean'].values
f2 = df1['bill_mean'].values
plt.scatter(f1, f2, c=df1['hcluster'], s=7)


# In[ ]:




