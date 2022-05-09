#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Part-1: Data Exploration and Pre-processing

# In[2]:


cd D:\FT\python\Ml\Assignment


# In[20]:


# 1) load the given dataset
data=pd.read_csv("Python_Project_4_Logistic.csv")


# In[21]:


# 2) print all the column names 
data.columns


# In[6]:


# 3) describe the data
data.describe()


# In[22]:


# 4) check the null value 
data.isnull().sum()


# In[23]:


# 5) if there are Null values, Handle these
data.dropna(inplace=True)
data.isnull().sum()


# Part-2: Working with Models 

# In[24]:


# 1) Create the target data and feature data where target data is survived
x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data.Survived


# In[25]:


x


# In[26]:


y


# In[27]:


x.dtypes


# In[14]:


# x = x.drop(['Cabin','Embarked','Sex'],axis=1)


# In[ ]:


data.drop(['Ticket'],axis=1,inplace=True)


# In[ ]:


data.dtypes


# In[ ]:


data.drop('Name',axis=1,inplace=True)


# In[28]:


# Label encoding
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()
x['Sex']=label.fit_transform(data['Sex'])
x['Cabin']=label.fit_transform(data['Cabin'])
x['Embarked']=label.fit_transform(data['Embarked'])


# In[29]:


x.dtypes


# In[30]:


# 2) Split the data into Training and testing Set 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[31]:


x_train


# In[32]:


x_test


# In[36]:


# 3) Create a Logistic regression model for Target and featur
# model=LogisticRegression()
model = LogisticRegression(solver='liblinear')


# In[37]:


model.fit(x_train,y_train)


# In[35]:


get_ipython().system('pip install scikit-learn  -U')


# In[39]:


data.shape


# In[ ]:


model = LogisticRegression(solver='liblinear')


# In[ ]:


predicted_value=model.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,f1_score


# In[ ]:


# 4) Display the Confusion Matrix
confusion_matrix(predicted_value,y_test)


# In[ ]:


# 5) Find the Accuracy Score
accuracy_score(predicted_value,y_test)


# In[ ]:


# 6) Find the Precision Score 
precision_score(predicted_value,y_test)


# In[ ]:


# 7) Find the Recall Score
recall_score(predicted_value,y_test)


# In[ ]:


#  8) Find the F1 Score
f1_score(predicted_value,y_test)


# In[ ]:


# 9) Find the probability of testing data 
model.predict_proba(x_test)


# In[ ]:


# 10) Display ROC Curve and find the AUC score

