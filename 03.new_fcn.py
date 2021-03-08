
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:18:14 2020

@author: hpg
"""
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Input,Dropout,BatchNormalization
import numpy as np

import tensorflow as tf
train, test = tf.keras.datasets.mnist.load_data()
trainimg, trainlabel = train
testimg, testlabel = test

trainlabel = tf.keras.utils.to_categorical(trainlabel, num_classes=10)
testlabel = tf.keras.utils.to_categorical(testlabel, num_classes=10)

batch_size=128
trainimg=np.load('trainlabel_oi_pb_arr.npy').astype(int)
testimg=np.load('testlabel_oi_pb_arr.npy').astype(int)

trainimg=(trainimg-trainimg.mean())/trainimg.var()
testimg=(testimg-testimg.mean())/testimg.var()

input = Input(shape=(testimg.shape[-1],))
x = Dense(128, activation='relu')(input)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
#x = Dense(16, activation='relu')(x)
#x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=input, outputs=output)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
epochs=500
model.fit(trainimg,trainlabel,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(testimg,testlabel),
              verbose=2)
loss,acc=model.evaluate(testimg,testlabel)
print('loss:',loss,'acc:',acc)

