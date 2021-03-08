# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:45:22 2020

@author: hpg
"""
import numpy as np
#_epsilon1
def epsilon1(f=0.5,p=0.5,q=0.75,h=2):
    q_=0.5*f*(p+q)+(1-f)*q
    p_=0.5*f*(p+q)+(1-f)*p
    _epsilon1=h*np.log((q_*(1-p_))/(p_*(1-q_)))
    return _epsilon1
#_epsilon0 目标 0.5 1 2 5
def epsilon0(h,f):
    _epsilon0=2*h*np.log((1-0.5*f)/(0.5*f))
    return _epsilon0
    
def epsilon(f,p,q,h):
    _epsilon=epsilon0(h,f)+epsilon1(f,p,q,h)
    return _epsilon

#f=0.5;p=0.618;q=0.75;h=2;#e0=>4.39,e1=>0.61,e=>5
f=0.5;p=0.995;q=0.75;h=2;#e0=>4.39,e1=>-2.37,e=>2  
f=0.5;p=0.358;q=0.75;h=2;#e0=>4.39,e1=>1.6,e=>6
f=0.5;p=0.55;q=0.75;h=2;#e0=>4.39,e1=>1.11,e=>5.5
f=0.5;p=0.05;q=0.75;h=2;#e0=>4.39,e1=>1.11,e=>5.5
f=0.95;p=0.07;q=0.18;h=2;#e0=>4.39,e1=>1.11,e=>5.5
#f=0.937;p=0.5;q=0.75;h=2;#e0=>0.5  acc=96.78 epoch=34
#f=0.875;p=0.5;q=0.75;h=2;#e0=>1    acc=96.96 epoch=30
#f=0.755;p=0.5;q=0.75;h=2;#e0=>2    acc=96.84 epoch=33
#f=0.445;p=0.5;q=0.75;h=2;#e0=>5    acc=96.2  epoch=28
print(epsilon0(h,f))
print(epsilon1(f,p,q,h))
print(epsilon(f,p,q,h))
farr=[i/100 for i in range(1,99)]
epsarr0=[epsilon0(h,f) for f in farr]
epsarr1=[epsilon1(h,p,q,f) for f in farr]

from matplotlib import pyplot as plt

plt.plot()
#plt.plot(farr,epsarr0)
plt.plot(farr,epsarr1)
plt.title('epsilon0=func(f)')
plt.ylabel('epsilon0')
plt.xlabel('f')
plt.show()
#print(epsilon1(f,p,q,h))
#print(epsilon(f,p,q,h))
#f=0.5;p=0.5;q=0.75;h=2
# f=0.757;p=0.5;q=0.51;h=2
'''1.983679352293049
0.019441982481271783
2.003121334774321
'''
'''
f=0.876;p=0.5;q=0.51;h=1
0.49856587903449
0.004960498593297921
0.503526377627788
'''
'''
f=0.757;p=0.5;q=0.51;h=1
0.9918396761465245
0.009720991240635891
1.0015606673871604
===========================================================
f=0.222;p=0.5;q=0.75;h=1
4.161134068403141
0.8445357304797941
5.005669798882935
-----------------------------------
f=0.62;p=0.5;q=0.75;h=1
1.6002386002242261
0.40699506080196796
2.007233661026194
acc:96.82  epoch:23
-----------------------------------
f=0.803;p=0.5;q=0.75;h=1
0.7984379832224229
0.21036341123867036
1.0088013944610932
acc:97.16  epoch:27
-----------------------------------
f=0.901;p=0.5;q=0.75;h=1
0.3973013935905667
0.10562914981195262
0.5029305434025193
acc:96.79  epoch:18
'''



#import sys;sys.exit()
testlabel_o=np.load('testlabel_oi_pb.npy')
testlabel_o=(testlabel_o-testlabel_o.min())/(testlabel_o.max()-testlabel_o.min())
testlabel_o=(testlabel_o*1000).astype(np.int)

trainlabel_o=np.load('trainlabel_oi_pb.npy')
trainlabel_o=(trainlabel_o-trainlabel_o.min())/(trainlabel_o.max()-trainlabel_o.min())
trainlabel_o=(trainlabel_o*1000).astype(np.int)
from collections import Counter
counter_testlabel_o=np.bincount(np.array(testlabel_o.flatten()))
counter_trainlabel_o=np.bincount(np.array(trainlabel_o.flatten()))
print(counter_testlabel_o)
print(counter_trainlabel_o)


trainimg=np.load('testlabel_o.npy')
testimg=np.load('trainlabel_o.npy')
trainimg=(trainimg-trainimg.min())/(trainimg.max()-trainimg.min())
testimg=(testimg-testimg.min())/(testimg.max()-testimg.min())
trainimg=(trainimg*1000).astype(np.int)
testimg=(testimg*1000).astype(np.int)
counter_trainimg=np.bincount(np.array(trainimg.flatten()))
counter_testimg=np.bincount(np.array(testimg.flatten()))

print(counter_trainimg)
print(counter_testimg)

x=[i for i in range(1001)]
import matplotlib.pyplot as plt
plt.plot(x,counter_testlabel_o)
plt.legend()
plt.title('test epslion')
plt.xlabel('value')
plt.ylabel('frequence')
plt.show()

plt.plot(x,counter_trainlabel_o)
plt.legend()
plt.title('train epslion')
plt.xlabel('value')
plt.ylabel('frequence')
plt.show()

plt.plot(x,counter_trainimg)
plt.legend()
plt.title('train')
plt.xlabel('value')
plt.ylabel('frequence')
plt.show()

plt.plot(x,counter_testimg)
plt.legend()
plt.title('test')
plt.xlabel('value')
plt.ylabel('frequence')
plt.show()