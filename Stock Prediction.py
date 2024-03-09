
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf


start = '2010-01-01'
end = '2019-12-31'
stock = 'AAPL'

df = yf.download(stock,start,end)

df

df.head()

df.tail()


df = df.reset_index()  
df.head()





df = df.drop(['Date','Adj Close'],axis = 1)
df.head()


# In[10]:


plt.plot(df.Close) #i want to predict closing price fo  r particuler date


# In[11]:


df


# In[12]:


ma100 = df.Close.rolling(100).mean()  # from 101th value it will show mean of previous value
ma100    #moving average
# for 100 value it will show Nan bcs no previous 100 value to get mean


# In[13]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
 


# In[14]:


ma200=df.Close.rolling(200).mean()
ma200


# In[15]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')


# In[16]:


df.shape


# In[17]:


# splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #till 70% of value
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
                                        
print(data_training.shape)
print(data_testing.shape)


# In[18]:


data_training.head()


# In[19]:


data_testing.head()


# In[20]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# In[21]:


data_training_array = scaler.fit_transform(data_training)
data_training_array


# In[22]:


data_training_array.shape


# In[65]:


x_train =[]
y_train= []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100 : i]) # i-100 means 0 (100-100)
    y_train.append(data_training_array[i,0]) # only one column
    
x_train, y_train = np.array(x_train), np.array(y_train) # convert a and y train in array


# In[66]:


x_train.shape


#   # ML Model

# In[67]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')
import keras.models
import tensorflow


# In[68]:


from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential


# In[69]:


model = Sequential()
model.add(LSTM(units = 50, activation ='relu', return_sequences = True,
              input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))



model.add(LSTM(units = 60, activation ='relu', return_sequences = True))
model.add(Dropout(0.3))



model.add(LSTM(units = 80, activation ='relu', return_sequences = True))
model.add(Dropout(0.4))



model.add(LSTM(units = 120, activation ='relu'))
model.add(Dropout(0.5))

#connect all layers
model.add(Dense(units = 1))


# In[70]:


model.summary()


# In[71]:


model.compile(optimizer='adam' , loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)


# In[72]:


model.save('keras_model.h5')


# In[73]:


data_testing.head()


# In[59]:


data_testing.head()


# In[74]:


past_100_days = data_training.tail(100) #bcs we want past 100 days data to test further more


# In[75]:


final_df = past_100_days._append(data_testing,ignore_index=True)


# In[76]:


final_df.head() #data not sacled bcs this is testing data


# In[77]:


input_data = scaler.fit_transform(final_df)
input_data


# In[78]:


input_data.shape


# In[79]:


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])


# In[80]:


x_test ,y_test = np.array(x_test) , np.array(y_test)
print(x_test.shape)
print(y_test.shape)


# In[81]:


#making predictions

y_predicted = model.predict(x_test)


# In[82]:


y_predicted.shape


# In[83]:


y_test


# In[84]:


y_predicted


# In[85]:


scaler.scale_  #by this all are scaled down


# In[86]:


scale_factor = 1/0.02123255
y_predicted = y_predicted * scale_factor
y_test= y_test * scale_factor


# In[87]:


plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




