import numpy as np
from decimal import Decimal
def dTob3(n, pre=16):
    is_positive_integer=True
    if n<0:
        n=-n
        is_positive_integer=False
    string_number1 = str(n)
    flag = False 
    if '.' in string_number1:
        flag = True
    resutl=0
    if flag:
        string_integer, string_decimal = string_number1.split('.') 
        integer = int(string_integer)
        decimal = Decimal(str(n)) - integer
        l2 = []
        decimal_convert = ""
        while True:
            if integer == 0: 
                break
            x,y = divmod(integer, 2)
            l2.append(y)
            integer = x
        string_integer = '-'.join([str(j) for j in l2[::-1]]) 
        i = 0
        while decimal != 0 and i < pre: 
            result = int(decimal * 2) 
            decimal = decimal * 2 - result 
            decimal_convert = decimal_convert + '-'+str(result) 
            i = i + 1
        resutl = string_integer + '.' + decimal_convert
    else:
        l2 = []
        while True: 
            if n == 0: break
            x,y = divmod(n, 2)
            l2.append(y)
            n = x
        resutl = '-'.join([str(j) for j in l2[::-1]]) 
    if is_positive_integer:
        resutl='0-'+str(resutl)
    else:
        resutl='1-'+str(resutl)
    return resutl.replace('.','').replace('-','')
import rappor
params = rappor.Params()
params.prob_f = 0.5
params.prob_p = 0.5
params.prob_q = 0.75
# return these 3 probabilities in sequence.
class MockRandom(object):
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

trainlabel_pb=np.load('trainlabel_pb.npy')
trainlabel_p=np.load('trainlabel_p.npy')
trainlabel_o=np.load('trainlabel_o.npy')
trainlabel_oi_pb=np.load('trainlabel_oi_pb.npy')
testlabel_pb=np.load('testlabel_pb.npy')
testlabel_p=np.load('testlabel_p.npy')
testlabel_o=np.load('testlabel_o.npy')
testlabel_oi_pb=np.load('testlabel_oi_pb.npy')

print('train orgin:',trainlabel_o[0])
print('train io_pb:',trainlabel_oi_pb[0])



import rappor  # module under test
print('train orgin:',trainlabel_o[0,:5])
print('train std:',trainlabel_p[0,:5])
for tmp in range(trainlabel_o[0,:5].shape[0]):
    print('std:{},rappor:{}'.format(tmp,REncode.encode(dTob3(tmp))))
print('train rappor:',trainlabel_pb[0,:5])
print('train rappor:',trainlabel_pb[0,:5])
print('test orgin:',testlabel_o[0,:5])
print('test std:',testlabel_p[0,:5])
for tmp in range(testlabel_o[0,:5].shape[0]):
    print('std:{},rappor:{}'.format(tmp,REncode.encode(dTob3(tmp))))
print('test rappor:',testlabel_pb[0,:5])
