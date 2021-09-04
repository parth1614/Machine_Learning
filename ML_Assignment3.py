#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import os
dataframe = pd.read_csv("Datasets\\AWCustomers.csv")
dataframe.info()


# In[27]:


dataframe['BirthDate'] = pd.to_datetime(dataframe['BirthDate'])

import datetime
CURRENT_TIME = datetime.datetime.now()
def get_age(birth_date, today = CURRENT_TIME):
    y = today-birth_date
    return y.days//365

dataframe['Age'] = dataframe['BirthDate'].apply(lambda x: get_age(x))
dataframe.drop(['BirthDate'], axis = 1, inplace  = True)
dataframe.info()


# In[28]:


df1 = pd.DataFrame(data = dataframe)
df2 = df1.loc[:, ['CustomerID', 'Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome', 'YearlyIncome', 'Age']]


# In[29]:


df2.head()


# In[30]:


df2.info()


# In[31]:


df2['Education'].value_counts()


# In[32]:


df2['Education'] = df2['Education'].map({'Partial High School' : 1, 'High School' : 2, 'Partial College' : 3, 'Bachelors' : 4, 'Graduate Degree' : 5})


# In[33]:


df2['Occupation'].value_counts()


# In[34]:


df2['Occupation'] = df2['Occupation'].map({'Manual' : 1, 'Skilled Manual' : 2, 'Clerical' : 3, 'Management' : 4, 'Professional' : 5})


# In[35]:


def cardinal_columns(df):
    df['Gender'] = df['Gender'].map({'M' : 1, 'F' : 0})
    df['MaritalStatus'] = df['MaritalStatus'].map({'M' : 1, 'S' : 0})
    return df

df2 = cardinal_columns(df2)
df2.head()


# In[36]:


df2.isnull().sum()


# In[37]:


from sklearn.preprocessing import MinMaxScaler
def scale_down(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['YearlyIncome', 'Age']])
    df['YearlyIncomeScaled'] = scaled[:, 0] 
    df['AgeScaled'] = scaled[:, 1]
    df.drop(['YearlyIncome', 'Age'], axis = 1, inplace = True)
    return df


# In[38]:


df2 = scale_down(df2)


# In[40]:


from scipy.spatial import distance
distance.cosine(df2['Education'].values, df2['YearlyIncomeScaled'].values)


# In[ ]:


distance.jaccard(df2['Education'].values, df2['YearlyIncomeScaled'].values)


# In[41]:


from scipy.stats import pearsonr
pearsonr(df2['Education'].values, df2['YearlyIncomeScaled'].values)[0]

