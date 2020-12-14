# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 20:27:28 2016

@author: xinruyue
"""

import random
import networkx as nx
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
        s = random.choice(n)
# random.randint(len(n))
        for j in range(10):
            nei = list(g.neighbors(s))
            next_node = random.choice(nei)
            seq.append(str(next_node))
            s = next_node
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

    #根据索引填充48*48矩阵
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

#二维数据栅格化并记录位置
def img_max_pos(node_vec,size):
    lowdim = []
    for each in node_vec:
        lowdim.append(each[0])

    #确定二维数据的x，y范围，分为48个
    xmax = np.amax(lowdim, axis=0)[0]
    ymax = np.amax(lowdim, axis=0)[1]
    xmin = np.amin(lowdim, axis=0)[0]
    ymin = np.amin(lowdim, axis=0)[1]

    x_scale = np.linspace(xmin, xmax, size)
    y_scale = np.linspace(ymin, ymax, size)

    #生成一个空48*48矩阵
    new_mat = np.zeros((size, size))

    pos_info = []
    #根据索引填充48*48矩阵
    for ind, each in enumerate(lowdim):
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

        new_mat[i,size-1-j] += 1
        node_no = node_vec[ind][1]
        pos_info.append(([i,size-1-j],node_no))

    return new_mat,pos_info


def imconv(image_array,filt):
    #计算卷积，原图像与算子卷积后的结果矩阵
    #input:
        #image_array:原始图片矩阵
        #filter:filter数据
    #return：卷积之后的图片矩阵
    dim1,_ = image_array.shape
    dim2,_ = filt.shape
    size = dim1 - dim2 + 1
    image = np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            image[i,j] = (image_array[(i):(i+5),(j):(j+5)]*filt).sum()
    return image

def order_conv1(im):
    #将conv1的值排序
    #pos_lists:图片的像素点位置以及对应的值 [([],value1),([],value2)...]
    #input：
        #im:卷积计算之后的图片矩阵
    #return：排序之后的pos_lists
    pos_lists = []
    d1,d2 = im.shape
    for i in range(d1):
        for j in range(d2):
            pos_lists.append(([i,j],im[i,j]))
    pos_lists.sort(key=lambda x:x[1],reverse=True)
    return pos_lists

def get_nodes(pos_lists,img_data,p_inf,f_size):
    #input：
        #post_lists:排序之后的图片的像素带你位置及对应的值
        #img_data:原图片矩阵
        #p_inf:每一个节点及在图片上的位置
        #f_size:filter的size
    #得到对应的原图的位置
    node_rec = []
    for i in range(1):
        max_value = pos_lists[i]
        max_pos = max_value[0]
        p1 = max_pos[0]
        p2 = max_pos[1]
    #    data_zone = img_data[p1:p1+f_size,p2:p2+f_size]
        #v_pos:有值的像素点的位置 [[],[],...]
        v_pos = []
        for i in range(f_size):
            for j in range(f_size):
                if img_data[p1+i,p2+j] > 0:
                    v_pos.append([p1+i,p2+j])
        #node_rec:有值的像素点对应的节点
        for each_v in v_pos:
            for each_p in p_inf:
                if each_p[0] == each_v:
                    node_rec.append(each_p[1])
    return node_rec
