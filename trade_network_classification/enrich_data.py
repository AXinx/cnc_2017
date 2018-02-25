#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:18:42 2017

@author: xinruyue
"""

import os
import pickle
import numpy as np
import random

#将两张图整合成一张图来增加数据量

f_path = './img_data'
f = os.listdir(f_path)

print(f)

for each in f[5:6]:

    label = each.split('_')[1]
    label = label.split('.')[0]
    print(label)

    #load data
    pkl_file = os.path.join(f_path,each)
    f1 = open(pkl_file,'rb')
    mat_data = pickle.load(f1)

    print(len(mat_data))

    img_data = []
    for e_mat in mat_data:
        non_zero = np.nonzero(e_mat)
        num_nonzero = len(non_zero[0])
        if num_nonzero >= 30:
            img_data.append(e_mat)

    s = len(mat_data)
    for n in range(8000):
        i = random.randint(0,(s-1))
        j = random.randint(0,(s-1))

        if i != j:
            new_data = mat_data[i] + mat_data[j]
            n_zero = np.nonzero(new_data)
            num_nzero = len(n_zero[0])
            if num_nzero >= 30:
                img_data.append(new_data)
        n += 1
    size = len(img_data)
    print(size)

    t_size = int(size * 0.8)
    v_size = int(size * 0.9)

    print(t_size)
        #train data and test data
    train_data = img_data[:t_size]
    val_data = img_data[t_size:v_size]
    test_data = img_data[v_size:]

    f2 = open('train_data_'+str(label)+'.pkl','wb')
    pickle.dump(train_data,f2)
    f2.close()

    f3 = open('test_data_'+str(label)+'.pkl','wb')
    pickle.dump(test_data,f3)
    f3.close()

    f4 = open('val_data_'+str(label)+'.pkl','wb')
    pickle.dump(val_data,f4)
    f4.close()

