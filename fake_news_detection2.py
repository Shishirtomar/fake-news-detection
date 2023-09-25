#!/usr/bin/env python
# coding: utf-8

# ## Fake news detection2
# 

# In[26]:


data = pd.read_csv("fake_or_real_news.csv")


# In[25]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  #text to convert into numbers as ML models only work on numbers.

from sklearn.svm import LinearSVC


# In[27]:


data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)


# In[28]:


data


# In[29]:


data = data.drop("label",axis =1)


# In[30]:


X, y = data['text'], data ['fake']


# In[31]:


X


# In[32]:


y


# In[33]:


X_train, X_test ,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[34]:


X_train


# In[35]:


len(X_train)


# In[36]:


len(X_test)


# In[37]:


vectorizer = TfidfVectorizer(stop_words = "english",max_df = 0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized  = vectorizer.transform(X_test)


# In[38]:


clf = LinearSVC()
clf.fit(X_train_vectorized,y_train)


# In[39]:


clf.score(X_test_vectorized,y_test)


# In[40]:


len(y_test) * 0.9479


# In[41]:


len(y_test)


# In[42]:


with open("mytext.txt","w",encoding = "utf-8") as f:
    f.write(X_test.iloc[10])


# In[43]:


with open("mytext.txt","r",encoding = "utf-8") as f:
    text = f.read()


# In[44]:


text


# In[47]:


vectorized_text = vectorizer.transform([text])


# In[48]:


clf.predict(vectorized_text)


# In[49]:


y_test.iloc[10]  #1:means fake news
                 #0:means true news


# In[ ]:




