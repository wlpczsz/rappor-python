# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 18:58:25 2020

@author: hpg
"""
import pandas as pd
import numpy as np
def epsilon1(f=0.5,p=0.5,q=0.75,h=2):
    q_=0.5*f*(p+q)+(1-f)*q
    p_=0.5*f*(p+q)+(1-f)*p
    _epsilon1=h*np.log((q_*(1-p_))/(p_*(1-q_)))
    return _epsilon1
def epsilon0(h,f):
    _epsilon0=2*h*np.log((1-0.5*f)/(0.5*f))
    return _epsilon0
    
def epsilon(f,p,q,h):
    _epsilon0=epsilon0(h,f)
    _epsilon1=epsilon1(f,p,q,h)
    _epsilon=_epsilon0+_epsilon1
    return _epsilon,_epsilon0,_epsilon1

f_arr=[i/10000 for i in range(1,10000)]
q_arr=[0.75]#[i/100 for i in range(25,50)]
p_arr=[0.5]#[i/100 for i in range(75-25,75+24)]
arr=[]
h=2
def get_epsilon0(f_arr,p_arr,q_arr):
    times=0
    old_time=0
    for f in f_arr:
        for p in p_arr:
            for q in q_arr:
                _epsilon,_epsilon0,_epsilon1=epsilon(f,p,q,h)
                dif=_epsilon0-int(_epsilon0)
                old_time=times
                if dif<=0.01 and _epsilon0 >0.8 and _epsilon0<10.5:
                    times+=1
                    arr.append({'f':f,'p':p,'q':q,'h':h,'epsilon0':_epsilon0,'epsilon1':_epsilon1,'epsilon':_epsilon})
                elif abs(_epsilon0-0.5)<=0.01 and _epsilon0<1:
                    times+=1
                    arr.append({'f':f,'p':p,'q':q,'h':h,'epsilon0':_epsilon0,'epsilon1':_epsilon1,'epsilon':_epsilon})
                if times%1==0 and old_time!=times:
                    print(times)
                    df = pd.DataFrame(arr)
                    df.to_csv('fqphe3.csv')
get_epsilon0(f_arr,p_arr,q_arr)
def get_epsilon(f_arr,p_arr,q_arr):
    times=0
    old_time=0
    for f in f_arr:
        for p in p_arr:
            for q in q_arr:
                _epsilon,_epsilon0,_epsilon1=epsilon(f,p,q,h)
                dif=_epsilon-int(_epsilon)
                old_time=times
                if _epsilon0*_epsilon1<0:
                    continue
                if dif<=0.01 and _epsilon >0.8 and _epsilon<10.5:
                    times+=1
                    arr.append({'f':f,'p':p,'q':q,'h':h,'epsilon0':_epsilon0,'epsilon1':_epsilon1,'epsilon':_epsilon})
                elif abs(_epsilon-0.5)<=0.01 and _epsilon<1:
                    times+=1
                    arr.append({'f':f,'p':p,'q':q,'h':h,'epsilon0':_epsilon0,'epsilon1':_epsilon1,'epsilon':_epsilon})
                if times%100==0 and old_time!=times:
                    print(times)
                    df = pd.DataFrame(arr)
                    df.to_csv('fqphe3.csv')