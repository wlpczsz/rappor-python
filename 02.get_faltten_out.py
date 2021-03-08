# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:08:12 2020
@author: hpg
"""
import numpy as np
#print()
#import sys;sys.exit()
from keras.models import load_model
from keras import Model
import tensorflow as tf
train, test = tf.keras.datasets.mnist.load_data()
trainimg, trainlabel = train
testimg, testlabel = test

trainimg=trainimg.reshape(trainimg.shape[0],28,28,1)
testimg=testimg.reshape(testimg.shape[0],28,28,1)
trainlabel = tf.keras.utils.to_categorical(trainlabel, num_classes=10)
testlabel = tf.keras.utils.to_categorical(testlabel, num_classes=10)

model=load_model('mnist.h5')
loss,acc=model.evaluate(testimg,testlabel)
print('loss:',loss,'acc:',acc)
#model.summary()
model2= Model(inputs=model.input, outputs=model.layers[-3].output)
model2.summary()

trainlabel_o=model2.predict(trainimg)
np.save('trainlabel_o',trainlabel_o)
print('trainlabel_o--[:2]:',trainlabel_o[:2])
trainlabel_p=(trainlabel_o-trainlabel_o.mean())/trainlabel_o.std()
#print('trainlabel_p[:2]:',trainlabel_p[:2])

testlabel_o=model2.predict(testimg)
np.save('testlabel_o',testlabel_o)
print('testlabel_o--[:2]:',testlabel_o[:2])
testlabel_p=(testlabel_o-testlabel_o.mean())/testlabel_o.std()
#print('testlabel_p[:2]:',testlabel_p[:2])

np.save('trainlabel_p',trainlabel_p)
np.save('testlabel_p',testlabel_p)

np.save('trainlabel_o',trainlabel_o)
np.save('testlabel_o',testlabel_o)

bit_len=32
import rappor  # module under test
params = rappor.Params()
params.num_bloombits=32
params.prob_f = 0.5
params.prob_p = 0.5
params.prob_q = 0.75
# return these 3 probabilities in sequence.
class MockRandom(object):
  """Returns one of three random values in a cyclic manner.
  Mock random function that involves *some* state, as needed for tests that
  call randomness several times. This makes it difficult to deal exclusively
  with stubs for testing purposes.
  """

  def __init__(self, cycle, params):
    self.p_gen = MockRandomCall(params.prob_p, cycle, params.num_bloombits)
    self.q_gen = MockRandomCall(params.prob_q, cycle, params.num_bloombits)

class MockRandomCall:
  def __init__(self, prob, cycle, num_bits):
    self.cycle = cycle
    self.n = len(self.cycle)
    self.prob = prob
    self.num_bits = num_bits

  def __call__(self):
    counter = 0
    r = 0
    for i in range(0, self.num_bits):
      rand_val = self.cycle[counter]
      counter += 1
      counter %= self.n  # wrap around
      r |= ((rand_val < self.prob) << i)
    return r
rand = MockRandom([0.0, 0.6, 0.0], params)
REncode = rappor.Encoder(params, 0, 'secret', rand)

print('RAPPOR encode data:start test data')
testlabel_pb=np.zeros(testlabel_o.shape)
for i in range(testlabel_o.shape[0]):
    for j in range(testlabel_o.shape[1]):
        #print(testlabel_o[i,j])
        testlabel_pb[i,j]=REncode.encode_bits(int(testlabel_o[i,j]))
np.save('testlabel_oi_pb',testlabel_pb)

print('RAPPOR encode data:start train data')
trainlabel_pb=np.zeros(trainlabel_o.shape)
for i in range(trainlabel_o.shape[0]):
    for j in range(trainlabel_o.shape[1]):
        trainlabel_pb[i,j]=REncode.encode_bits(int(trainlabel_o[i,j]))
np.save('trainlabel_oi_pb',trainlabel_pb)
print(trainlabel_pb.max())


bits_len=bit_len
trainlabel_o=np.load('trainlabel_oi_pb.npy')
trainlabel_o_arr=np.zeros((60000,32,bits_len))
for i in range(trainlabel_o.shape[0]):
    for j in range(trainlabel_o.shape[1]):
        s='{0:b}'.format(int(trainlabel_o[i,j]))
        while len(s)<bits_len:
            s='0'+s
        for k in range(len(s)):
            trainlabel_o_arr[i,j,k]=int(s[k])
trainlabel_o_arr=trainlabel_o_arr.reshape((60000,-1)).astype(np.int)
print(trainlabel_o_arr.shape)
np.save('trainlabel_oi_pb_arr',trainlabel_o_arr)

testlabel_o=np.load('testlabel_oi_pb.npy')
testlabel_o_arr=np.zeros((10000,32,bits_len))
for i in range(testlabel_o.shape[0]):
    for j in range(testlabel_o.shape[1]):
        s='{0:b}'.format(int(testlabel_o[i,j]))
        while len(s)<bits_len:
            s='0'+s
        for k in range(len(s)):
            testlabel_o_arr[i,j,k]=int(s[k])
testlabel_o_arr=testlabel_o_arr.reshape((10000,-1)).astype(np.int)
print(testlabel_o_arr.shape)
np.save('testlabel_oi_pb_arr',testlabel_o_arr)









