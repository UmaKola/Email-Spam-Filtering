#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[3]:


spam_df = pd.read_csv("spam.csv")


# In[4]:


spam_df


# In[7]:


#group by the category i.e., spam or ham and describe like how many ham and spam are there and how many unique
spam_df.groupby('Category').describe()


# In[8]:


#creating a spam column and categorizing them if spam 0 else 1
spam_df['spam'] = spam_df['Category'].apply(lambda x : 1 if x == "spam" else 0)


# In[9]:


spam_df


# In[13]:


#create train_test_split
#x be the content of my mails and y is the label i.e.,zero or one
x_train, x_test , y_train , y_test = train_test_split(spam_df.Message, spam_df.spam , test_size = 0.25)


# In[14]:


x_train


# In[15]:


x_train.describe()


# In[16]:


#word count and store this data as numerical matrix
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)


# In[17]:


x_train_count


# In[18]:


x_train_count.toarray()


# In[21]:


#train model
model = MultinomialNB()
model.fit(x_train_count, y_train)


# In[23]:


#prestest ham
email_ham = ["hey wanna meet up for the game"]
email_ham_count = cv.transform(email_ham)
model.predict(email_ham_count)


# In[26]:


#pre-test spam
email_spam = ["click here for the reward"]
email_spam_count = cv.transform(email_spam)
model.predict(email_spam_count)


# In[28]:


#test_model
x_test_count = cv.transform(x_test)
#give all the values
model.predict(x_test_count)


# In[29]:


#give the prediction
model.score(x_test_count , y_test)

