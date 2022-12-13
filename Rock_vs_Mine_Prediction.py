#!/usr/bin/env python
# coding: utf-8

# # Installing Libraries

# In[2]:


pip install scikit-learn


# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # LOADING FILE

# In[14]:


sonar_data = pd.read_csv('D:\ml projects\sonar data.csv',header= None)


# In[16]:


sonar_data.head()


# In[ ]:


#rowns and coloumns


# In[18]:


sonar_data.shape


# In[20]:


sonar_data.describe()


# In[23]:


sonar_data[60].value_counts()


# In[24]:


sonar_data.groupby(60).mean()


# In[26]:


X = sonar_data.drop(columns= 60)
Y = sonar_data[60]


# In[27]:


print(X)
print(Y)


# # Training and testing of data

# In[30]:


X_train ,  X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.1 , stratify=Y,random_state=1)


# In[32]:


print(X.shape , X_train.shape , X_test.shape)


# In[37]:


print(X_train)
print(Y_train)      


# # MODEL TRAINING 

# In[33]:


model = LogisticRegression()


# In[38]:


#Training the logistic regresssion model with training data


# In[39]:


model.fit(X_train,Y_train)


# # MODEL EVALUATION

# In[40]:


#accuaracy of model


# In[45]:


X_train_prediction =  model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[46]:


print('Accuracy for training data', training_data_accuracy)


# In[47]:


X_test_prediction =  model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[48]:


print('Accuracy for testing data', test_data_accuracy)


# # Making a predictive system

# In[56]:


input_data = (0.1083,0.1070,0.0257,0.0837,0.0748,0.1125,0.3322,0.4590,0.5526,0.5966,0.5304,0.2251,0.2402,0.2689,0.6646,0.6632,0.1674,0.0837,0.4331,0.8718,0.7992,0.3712,0.1703,0.1611,0.2086,0.2847,0.2211,0.6134,0.5807,0.6925,0.3825,0.4303,0.7791,0.8703,1.0000,0.9212,0.9386,0.9303,0.7314,0.4791,0.2087,0.2016,0.1669,0.2872,0.4374,0.3097,0.1578,0.0553,0.0334,0.0209,0.0172,0.0180,0.0110,0.0234,0.0276,0.0032,0.0084,0.0122,0.0082,0.0143)


#changing the input data into numpy array


input_data_into_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instace

input_data_reshaped = input_data_into_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]== 'R'):
    print("Object is Rock")
else:
    print("Object is Mine")

