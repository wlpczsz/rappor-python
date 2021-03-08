from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Flatten
from keras.layers.pooling import GlobalAveragePooling2D
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import tensorflow as tf
train, test = tf.keras.datasets.mnist.load_data()
trainimg, trainlabel = train
testimg, testlabel = test

trainimg=trainimg.reshape(trainimg.shape[0],28,28,1)
testimg=testimg.reshape(testimg.shape[0],28,28,1)
trainlabel = tf.keras.utils.to_categorical(trainlabel, num_classes=10)
testlabel = tf.keras.utils.to_categorical(testlabel, num_classes=10)
if True:
    base_num=16
    model = Sequential()
    model.add(Conv2D(base_num, (3, 3), input_shape=(28,28,1), padding='same', activation='relu'))
    model.add(Conv2D(base_num, (3, 3),  padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(2*base_num, (3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(2*base_num, (3, 3),  padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(2*base_num, (3, 3),  padding='same', activation='relu'))
    model.add(GlobalAveragePooling2D())#(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dense(10, activation='softmax'))#tf.nn.log_softmax
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
model.fit(trainimg,trainlabel, epochs=20,batch_size=1000,
          validation_data=(testimg,testlabel),
            verbose=2)
model.save('mnist.h5')
 
