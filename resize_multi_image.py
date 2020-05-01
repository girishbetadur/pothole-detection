# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:45:16 2020

@author: Harish
"""


import glob
import cv2
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


path1 = (r"C:\Users\Harish\Desktop\pothole_proj\potholes")
path2 = (r"C:\Users\Harish\Desktop\pothole_proj\normal")


pothole_img = glob.glob(path1+"/*.jpg")
normal_img = glob.glob(path2+"/*.jpg")

#path_normal_gray = r"C:\Users\Harish\Desktop\pothole_proj\normal_gray"
#path_pothole_gray = r"C:\Users\Harish\Desktop\pothole_proj\pothole_gray"

#path_normal_resize = r"C:\Users\Harish\Desktop\pothole_proj\normal_resize"

len_normal_img = len(normal_img)
len_pothole_img = len(pothole_img)

x_train = np.zeros((len_pothole_img + len_normal_img,50,50))
y_train_data=pd.read_csv("output.csv")
y_train=y_train_data.values

count = 0

for img in pothole_img:
    img1 = cv2.imread(img)
    #new_img = cv2.resize(img,(10,10))
    resize_img = cv2.resize(img1,(50,50))  
    BGR2gray = cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(os.path.join(path_pothole_gray,str(count)+".jpg"),BGR2gray)
    x_train[count,:,:] = BGR2gray
    count = count + 1
    
    
count = len_pothole_img


for img in normal_img:
    img1 = cv2.imread(img)
    gray_img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    resize_img = cv2.resize(img1,(50,50))  
    #cv2.imwrite(os.path.join(path_normal_resize,str(count)+".jpg"),resize_img)
    
    x_train[count,:,:] = resize_img
    print(count)
    count = count + 1
    
    
batch_size = 5
num_classes = 2
epochs = 5

x_train = x_train.reshape(len_pothole_img + len_normal_img,50,50,1)
y_train = keras.utils.to_categorical(y_train, num_classes)

cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(99,100,1)))
cnn.add(BatchNormalization())
#cnn.add(Dropout(0.3))
#cnn.add(Conv2D(32, (3, 3), activation='relu'))
#cnn.add(BatchNormalization())
#cnn.add(Dropout(0.3))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(BatchNormalization())
#cnn.add(Dropout(0.1))
cnn.add(MaxPooling2D(pool_size=(2,2),strides=2))
cnn.add(Flatten())

cnn.add(Dense(64, activation='relu'))
cnn.add(Dropout(0.1))
cnn.add(Dense(128,activation='relu'))
cnn.add(Dropout(0.1))
cnn.add(Dense(128,activation='relu'))

cnn.add(Dense(2, activation='sigmoid'))
cnn.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
cnn.fit(x_train, y_train,epochs=epochs,batch_size=batch_size,verbose=1,validation_split=0.05)
cnn.summary()
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

cnn.save("cnn_model.h5")
cnn.save("cnn_model.model")
    
