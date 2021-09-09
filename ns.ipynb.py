#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf


# In[2]:


data = pd.read_csv("Train_Data.csv")
data.head()


# In[3]:


print("shape = ",data.shape, "\n")
# df.info()


# In[4]:


data.describe()


# In[5]:


test_df = pd.read_csv('Test_Data.csv')

Xtest2 = test_df.drop(['date','ad'],axis=1)
col = ['campaign','adgroup','impressions','clicks','cost','conversions']  # ,'adnew','day','month'
Xtest2= Xtest2[col]
Xtest2.head(7)


# In[6]:


model =GradientBoostingRegressor(learning_rate=0.22)


# In[7]:


X = data.drop(['date','revenue','ad'],axis=1)
y = data.revenue


sc = StandardScaler()
# X = sc.fit_transform(X)
# Xtest2 = sc.transform(Xtest2)
cols =['impressions','clicks','cost','conversions']
X[cols] = sc.fit_transform(X[cols])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
X_train.head()


# In[8]:


Xtest2[cols] =sc.transform(Xtest2[cols])
Xtest2.head()


# In[9]:


columns_trans1 = make_column_transformer((OrdinalEncoder(),['campaign','adgroup']),remainder='passthrough')


# In[10]:


pipe2 = make_pipeline(columns_trans1,model)
pipe2.fit(X_train,y_train)


# In[11]:


print("MAE = ",mean_absolute_error(y_test,pipe2.predict(X_test)) )
print('r2 = ', r2_score(y_test,pipe2.predict(X_test)))


# In[12]:


res = pd.DataFrame(pipe2.predict(Xtest2),columns=['revenue'])
# res[res['revenue']<0] =0
res.to_csv('resulttemp.csv')
res.head(7)


# In[ ]:




