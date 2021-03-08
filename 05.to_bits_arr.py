import numpy as np
trainlabel_pb=np.load('trainlabel_pb.npy')
trainlabel_pb_arr=np.zeros((60000,32,8))
for i in range(trainlabel_pb.shape[0]):
    for j in range(trainlabel_pb.shape[1]):
        s='{0:b}'.format(int(trainlabel_pb[i,j]))
        for k in range(len(s)):
            trainlabel_pb_arr[i,j,k]=int(s[k])
        while k<trainlabel_pb_arr.shape[2]:
            trainlabel_pb_arr[i,j,k]=0
            k+=1
trainlabel_pb_arr=trainlabel_pb_arr.reshape((60000,-1)).astype(np.int)
print(trainlabel_pb_arr.shape)
np.save('trainlabel_pb_arr',trainlabel_pb_arr)

testlabel_pb=np.load('testlabel_pb.npy')
testlabel_pb_arr=np.zeros((10000,32,8))
for i in range(testlabel_pb.shape[0]):
    for j in range(testlabel_pb.shape[1]):
        s='{0:b}'.format(int(testlabel_pb[i,j]))
        for k in range(len(s)):
            testlabel_pb_arr[i,j,k]=int(s[k])
        while k<testlabel_pb_arr.shape[2]:
            testlabel_pb_arr[i,j,k]=0
            k+=1
testlabel_pb_arr=testlabel_pb_arr.reshape((10000,-1)).astype(np.int)
print(testlabel_pb_arr.shape)
np.save('testlabel_pb_arr',testlabel_pb_arr)

trainlabel_o=np.load('trainlabel_oi_pb.npy')
trainlabel_o_arr=np.zeros((60000,32,8))
for i in range(trainlabel_o.shape[0]):
    for j in range(trainlabel_o.shape[1]):
        s='{0:b}'.format(int(trainlabel_o[i,j]))
        for k in range(len(s)):
            trainlabel_o_arr[i,j,k]=int(s[k])
        while k<trainlabel_o_arr.shape[2]:
            trainlabel_o_arr[i,j,k]=0
            k+=1
trainlabel_o_arr=trainlabel_o_arr.reshape((60000,-1)).astype(np.int)
print(trainlabel_o_arr.shape)
np.save('trainlabel_oi_pb_arr',trainlabel_o_arr)

testlabel_pb=np.load('testlabel_oi_pb.npy')
testlabel_pb_arr=np.zeros((10000,32,8))
for i in range(testlabel_pb.shape[0]):
    for j in range(testlabel_pb.shape[1]):
        s='{0:b}'.format(int(testlabel_pb[i,j]))
        for k in range(len(s)):
            testlabel_pb_arr[i,j,k]=int(s[k])
        while k<testlabel_pb_arr.shape[2]:
            testlabel_pb_arr[i,j,k]=0
            k+=1
testlabel_pb_arr=testlabel_pb_arr.reshape((10000,-1)).astype(np.int)
print(testlabel_pb_arr.shape)
np.save('testlabel_pb_arr',testlabel_pb_arr)
