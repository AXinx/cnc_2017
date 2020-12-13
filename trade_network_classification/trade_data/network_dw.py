# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 20:27:28 2016

@author: xinruyue
"""

import random
#import networkx as nx
import numpy as np
from numpy import *
import gensim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def randomWalk(g,num):
    corpus = []
    n = g.nodes()
    for i in range(num):
        seq = []
        s = np.random.choice(n)
# random.randint(len(n))
        for j in range(10):
            nei = g.neighbors(s)
            if len(nei) == 0:
                continue
            next_node = random.choice(nei)
            seq.append(str(next_node))
            s = next_node
        corpus.append(seq)
    return corpus

# new randomwalk
def get_idx(col,s):
    s = sum(col)
    prob = []
    for each in col:
        prob.append(each/s)
    new_idx = np.random.choice(range(len(col)),1,p=prob)
    return new_idx

def random_walk(mat,num):
    size,_ = mat.shape
    corpus = []
    for i in range(num):
        p_idx = random.choice(range(size))
        p_col = mat[:,p_idx]
        global seq
        while not p_col.any():
            p_idx = random.choice(range(size))
            p_col = mat[:,p_idx]
        seq = []
        seq.append(str(p_idx))
        s = sum(p_col)
        new_idx = get_idx(p_col,s)
        while (len(seq)) < 10:
            new_col = mat[:,new_idx].reshape(size,)
            if new_col.any():
                s = sum(new_col)
                seq.append(str(new_idx.tolist()[0]))
                new_idx = get_idx(new_col,s)
            else:
                new_idx = get_idx(p_col,s)
            new_i = str(new_idx.tolist()[0])
            if new_i in seq:
                break
            if len(seq) == 1:
                break
        corpus.append(seq)
    return corpus

# word2vec
def word_2_vec(sequences, dim):
    model = gensim.models.Word2Vec(sequences, min_count=1, size=dim)

    nodes = model.wv.vocab.keys()
    em_vec = []
    for node in nodes:
        em_vec.append(model[node].tolist())
#   print np.mat(em_vec)
    return (np.mat(em_vec),model,nodes)

#二维数据栅格化
def img_max(lowdim,size):

    #确定二维数据的x，y范围，分为48个
    xmax = np.amax(lowdim, axis=0)[0]
    ymax = np.amax(lowdim, axis=0)[1]
    xmin = np.amin(lowdim, axis=0)[0]
    ymin = np.amin(lowdim, axis=0)[1]

    x_scale = np.linspace(xmin, xmax, size)
    y_scale = np.linspace(ymin, ymax, size)

    #生成一个空48*48矩阵
    new_mat = np.zeros((size, size))

    #根据索引填充128*128矩阵
    for each in lowdim:
        x = each[0]
        y = each[1]
        i = 0
        j = 0
        for idx, each in enumerate(x_scale):
            global i
            if idx < len(x_scale) - 1:
                if x >= each and x < x_scale[idx + 1]:
                    i = idx
            else:
                if x == each:
                    i = idx
        for idx, each in enumerate(y_scale):
            global j
            if idx < len(y_scale) - 1:
                if y > each and y < y_scale[idx + 1]:
                    j = idx
            else:
                if y == each:
                    j = idx

        new_mat[i][size - 1 - j] += 1

    return new_mat
