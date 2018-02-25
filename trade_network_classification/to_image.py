# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:14:31 2017

@author: Administrator
"""
from network_dw import *
import os
import pickle

def get_img(g):
    #2.deepWalk
    corp = random_walk(g,10000)
    vec,mod,nod = word_2_vec(corp,20)
    #3.pca
    pca = PCA(n_components=2)
    lowdim = pca.fit_transform(vec)
    #4.栅格化
    img_data = img_max(lowdim,48)
    return img_data

f_path = './matrix_data'
f = os.listdir(f_path)

for each in f[7:]:
    label = each.split('_')[1]
    label = label.split('.')[0]

    #load data
    pkl_file = os.path.join(f_path,each)
    f1 = open(pkl_file,'rb')
    all_matrix = pickle.load(f1,encoding='iso-8859-1')
    
    n = 0
    c = 0
    #generate img data for mat
    img_dt = []
    for e_mat in all_matrix:
#        if len(np.nonzero(e_mat)[0])>10:
        corp = random_walk(e_mat,10000)
        vec,mod,nod = word_2_vec(corp,20)

        pca = PCA(n_components=2)
        lowdim = pca.fit_transform(vec)

        if len(lowdim) > 1:
            c += 1
            print(c)
            img_data = img_max(lowdim,48)
            img_dt.append(img_data)
        n += 1
        print(n)
    
        
    f2 = open('imgdata_'+str(label)+'.pkl','wb')
    pickle.dump(img_dt,f2)
    f2.close()

