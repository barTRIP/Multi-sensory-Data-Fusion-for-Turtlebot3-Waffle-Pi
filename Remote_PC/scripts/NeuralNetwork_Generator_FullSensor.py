#! /usr/bin/env python3
"""
Created on Tue Oct 27 22:58:37 2020

@author: ajbc9
"""

from pandas import read_csv
#from pandas import write_csv

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adadelta
from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt
import numpy as np

#def root_mean_squared_error(y_true, y_pred):
    #return K.sqrt(mean_squared_error(y_true, y_pred))
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# load dataset
dataframe = read_csv("FullData.csv", delim_whitespace=False, header=None)
dataset = dataframe.values
print(dataset)

# Split the data into input (x) and ouput (y)
X = dataset[:,0:3]
Y = dataset[:,3]

# XN = np.array(X, dtype = float)
# YN = np.array(Y, dtype = float)
#
# # Reshaping the arrays into LSTM desired format (samples, timesteps, features)
# XN = np.reshape(XN, (1317, 1, 94))

LidarX = dataset[:, 0]
LidarY = Y
train=[]
test=[]
Epochs=150

# Splitting training and testing data, with training data being 80% of the data, and testing data being the remaining 20% of the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#LR = [500]

#for i in LR:
#Defines linear regression model and its structure
model = Sequential()
model.add(Dense(3, input_dim=3, kernel_initializer="he_uniform",activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(80, kernel_initializer="he_uniform", bias_initializer='zeros',activation='relu'))
model.add(Dense(1))

#Compiles model
model.compile(Adam(lr=0.008, decay=2/Epochs), loss="mean_squared_error", metrics =["mse"])

#Fits model
history = model.fit(np.array(X_train), np.array(y_train), batch_size=4, epochs = Epochs, shuffle=True, validation_data = (X_test, y_test), verbose = 0)
#Default batch size of 32
history_dict=history.history

#Plots model's training cost/loss and model's validation split cost/loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.figure()
plt.plot(loss_values,'bo',label='training loss') #Training data loss
plt.plot(val_loss_values,'r',label='val training loss') #Validation split loss
plt.title('Model Train Loss Lidar-Sonar-Camera')
plt.ylabel('loss function')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
model.summary()

y_train_pred = model.predict(X_train[:3000])
y_test_pred = model.predict(X[:3000]) #X_test
#y_test_pred = model.predict(X_test[:200]) #X_test

for i in range(0, len(y_train)):
    #print(i)
    if y_train[i]>3.5 and y_train_pred[i]>3.5:
        #print("training", i)
        train.append(i)
    if Y[i]>3.5 and y_test_pred[i]>3.5:
        #print("testing", i)
        test.append(i)
#Removing true and fused values above 3.5m as these count as "inf" and are considered the same.
y_train=np.delete(y_train,train)
y_train_pred=np.delete(y_train_pred, train)
Y=np.delete(Y, test)
y_test_pred=np.delete(y_test_pred, test)

#Printing in screen command line the RMSE
print("The RSME score on the Train set is:\t{:0.6f}".format(mean_squared_error(y_train, y_train_pred, squared=False)))
print("The RSME score on the Test set is:\t{:0.6f}".format(mean_squared_error(Y, y_test_pred, squared=False)))
print("The RSME score on the Lidar set is:\t{:0.6f}".format(mean_squared_error(LidarY, LidarX, squared=False)))

#Saving ANN model to callable file.
model.save("ANN_data_fusion23.h5")
print("Saved model to disk")

#GEnerating graphs.
fig, ax = plt.subplots(1, figsize=(8, 6))
fig.suptitle('ANN Testing Lidar-Sonar-Camera', fontsize=15)
plt.ylabel('Distance to obstacle [m]')
plt.xlabel('No. of measurement')
plt.plot(Y, 'bo',label='True distance')
plt.plot(y_test_pred,'r',label='Fused distance')
ax.plot(loc=1, title="Legend Title", frameon=False)
ax.legend(loc=1,bbox_to_anchor=(1.13,1.167),title="Legend", frameon=True)

plt.show()

fig, bx = plt.subplots(1, figsize=(8, 6))
fig.suptitle('ANN Training Lidar-Sonar-Camera ', fontsize=15)
plt.ylabel('Distance to obstacle [m]')
plt.xlabel('No. of measurement')
plt.plot(y_train, 'b',label='True distance')
plt.plot(y_train_pred,'r',label='Fused distance')
bx.plot(loc=1, title="Legend Title", frameon=False)
bx.legend(loc=1,bbox_to_anchor=(1.13,1.167),title="Legend", frameon=True)

plt.show()
