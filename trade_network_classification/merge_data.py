#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl
import os
import networkx as nx
#from network_dw import *

file_path = './pkl_file'
files = os.listdir(file_path)

pro_idx = {}
for i in range(10):
    f_name = []
    for each in files:
        s = each.split('_')[1]
        if s[0] == str(i):
            f_name.append(each)
    pro_idx[i] = f_name

for key,value in pro_idx.items():
    f_data = open('data_'+str(key)+'.pkl','wb')
    e_file = []
    for each in value:
        print each
        e_file_p = os.path.join(file_path,each)
        e_file_c = open(e_file_p,'rb')
        e_file.append(pkl.load(e_file_c))
    pkl.dump(e_file,f_data)
    f_data.close()