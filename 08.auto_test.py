from keras.models import Model
from keras.layers import Dense,Input,Dropout,BatchNormalization
import tensorflow as tf
train, test = tf.keras.datasets.mnist.load_data()
trainimg, trainlabel = train
testimg, testlabel = test
trainlabel = tf.keras.utils.to_categorical(trainlabel, num_classes=10)
testlabel = tf.keras.utils.to_categorical(testlabel, num_classes=10)

import numpy as np
bit_len=32
import rappor  # module under test
from rappor_test import MockRandom
def get_bin_arr(REncode,inarr,name,bits_len):
    nparr=np.zeros(inarr.shape)
    print('RAPPOR encode {}:start test data'.format(name))
    for i in range(nparr.shape[0]):
        for j in range(nparr.shape[1]):
            #print(testlabel_o[i,j])
            nparr[i,j]=REncode.encode_bits(int(inarr[i,j]))
    np.save('{}_oi_pb'.format(name),nparr)
    
    oi_bp_arr=np.zeros((nparr.shape[0],nparr.shape[1],bits_len))
    for i in range(nparr.shape[0]):
        for j in range(nparr.shape[1]):
            s='{0:b}'.format(int(nparr[i,j]))
            while len(s)<bits_len:
                s='0'+s
            for k in range(len(s)):
                oi_bp_arr[i,j,k]=int(s[k])
    np.save('{}_oi_pb_arr'.format(name),oi_bp_arr)
    oi_bp_arr=(oi_bp_arr-oi_bp_arr.mean())/oi_bp_arr.var()
    return oi_bp_arr
def get_model(testimg):
    inputs = Input(shape=(testimg.shape[-1],))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dense(16, activation='relu')(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
testlabel_o=np.load('testlabel_o.npy')
trainlabel_o=np.load('trainlabel_o.npy')

#get params list
import pandas as pd
df = pd.read_excel('fqphe.xlsx', sheet_name = '0.3,0.8all')
df['acc']=None
df['epoch']=-1
typicalNDict = {1:10,2:10,3:10,4:10,5:10,6:10,7:10,8:10,9:10,10:10}
def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)
df['epsilon']=df['epsilon'].astype(int)

#random choose 100 line date
result = df.groupby('epsilon').apply(typicalsamling, typicalNDict)
print(result)

for i in range(result.shape[0]):
    params = rappor.Params()
    params.num_bloombits=32
    params.prob_f = result['f'][i]
    params.prob_p = result['p'][i]
    params.prob_q = result['q'][i]
    rand = MockRandom([0.0, 0.6, 0.0], params)
    REncode = rappor.Encoder(params, 0, 'secret', rand)
    testlabel_oi_pb_arr=get_bin_arr(REncode,testlabel_o,'testlabel',bits_len=32)
    trainlabel_oi_pb_arr=get_bin_arr(REncode,trainlabel_o,'trainlabel',bits_len=32)
    
    from keras.callbacks import EarlyStopping,History
    history = History()
    early_stopping=EarlyStopping(monitor='val_acc', patience=1, verbose=2, mode='auto')
    callbacks = [
       history,
       early_stopping,
    ]
    print('get model')
    model=get_model(testlabel_oi_pb_arr)
    print('start tarin')
    model.fit(trainlabel_oi_pb_arr,trainlabel,
                  epochs=50,
                  batch_size=128,
                  validation_data=(testlabel_oi_pb_arr,testlabel),
                  verbose=2)
    print('evaluate model')
    loss,acc=model.evaluate(testimg,testlabel)
    result['acc'][i]=acc
    result['epoch'][i]=history.history['acc'].shape[0]
    print(result.loc[i])
    result.to_csv('auto_test.csv')
