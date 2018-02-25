#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:36:08 2017

@author: xinruyue
"""

import pickle
from scipy import misc
import matplotlib.pyplot as plt

'''
#open filter
f1 = open('w_conv1.pkl','rb')
data1 = pickle.load(f1)

#conv1
kernels = []
for i in range(data1.shape[3]):
    kernel = data1[:,:,:,i]
    kernels.append(kernel.reshape((5,5)))
n = 0
for each in kernels:
    plt.figure(n)
    plt.title('conv1_'+'filter_'+str(n),fontsize=20)
    plt.axis('off')
    plt.imshow(each,cmap=plt.cm.gray)
    plt.savefig('filter1_'+str(n)+'.eps')
    n += 1
#    misc.imsave('test.png',each)

'''

#open filter
f1 = open('w_conv2.pkl','rb')
data1 = pickle.load(f1)
#conv2
kernels = []
for i in range(data1.shape[3]):
    for j in range(data1.shape[2]):
        kernel = data1[:,:,j,i]
        kernels.append(kernel.reshape((5,5)))
print(len(kernels))

n = 0
for each in kernels:
    plt.figure(n)
    plt.axis('off')
    plt.title('conv2_'+'filter'+str(int(n/3))+'_'+str(n%3),fontsize=20)
    plt.imshow(each,cmap=plt.cm.gray)
    plt.savefig('filter2_'+str(n)+'.eps')
    n += 1
