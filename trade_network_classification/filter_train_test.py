# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:12:13 2017

@author: Administrator
"""
import os
import pickle
import numpy as np

f_path = './img_data'
f = os.listdir(f_path)

for each in f:

    label = each.split('_')[1]
    label = label.split('.')[0]
    print(label)

    #load data
    pkl_file = os.path.join(f_path,each)
    f1 = open(pkl_file,'rb')
    mat_data = pickle.load(f1)

    img_data = []
    for e_mat in mat_data:
        non_zero = np.nonzero(e_mat)
        num_nonzero = len(non_zero[0])
        if num_nonzero >= 30:
            img_data.append(e_mat)

    size = len(img_data)
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


'''
    # label file
    data_labels = np.zeros((size,10))
    data_labels[:, int(label) ] = 1

    train_label = data_labels[:t_size]
    test_label = data_labels[t_size:]

    f4 = open('train_label_'+str(label)+'.pkl','wb')
    pickle.dump(data_labels,f4)
    f4.close()

    f5 = open('test_label_'+str(label)+'.pkl','wb')
    pickle.dump(data_labels,f5)
    f5.close()
'''


