#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:46:31 2018

@author: xinruyue
"""

import sys
import os
import pickle as pkl
from network_dw import *

f_path = './shuffle_data'
f = os.listdir(f_path)

for each in f:
    label = each.split('_')[0]
    tp = label[1].split('.')[0]

    #load data
    pkl_file = os.path.join(f_path,each)
    f1 = open(pkl_file,'rb')
    all_matrix = pickle.load(f1,encoding='iso-8859-1')
    
    n = 0
    c = 0
    #generate img data for mat
    img_dt = []
    for e_mat in all_matrix:
        pca = PCA(n_components=2)
        lowdim = pca.fit_transform(vec)

        if len(lowdim) > 1:
            c += 1
            print(c)
            img_data = img_max(lowdim,48)
            img_dt.append(img_data)
        n += 1
        print(n)
    
        
    f2 = open('imgdata_'+tp+str(label)+'.pkl','wb')
    pickle.dump(img_dt,f2)
    f2.close()

