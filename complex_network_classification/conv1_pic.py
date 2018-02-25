

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:35:43 2017

@author: xinruyue
"""

import numpy as np
from PIL import Image
import pickle
from network_dw import *
from scipy.misc import imsave
from scipy import misc
import matplotlib.pyplot as plt
import pickle
from itertools import combinations

#1.无标度网
B = nx.random_graphs.barabasi_albert_graph(1000,4)
#小世界网
W = nx.random_graphs.watts_strogatz_graph(1000,8,0.1)

g = W

#2.deepWalk
corp = randomWalk(g,10000)
vec,mod,nod = word_2_vec(corp,20)

#3.pca
pca = PCA(n_components=2)
lowdim = pca.fit_transform(vec)

#4.栅格化及记录位置
#img_data:生成的图片矩阵
#p_inf:每个像素点对应的节点 [([],''),([],'')...]
node_vec = list(zip(lowdim,nod))
img_data,p_inf = img_max_pos(node_vec,48)

#5.打开filter
f1 = open('w_conv1.pkl','rb')
data1 = pickle.load(f1)
kernels = []
for i in range(data1.shape[3]):
    kernel = data1[:,:,:,i]
    kernels.append(kernel.reshape((5,5)))

size = len(kernels)
'''
#6.每个filter对应原图节点
short_path = []
for i in range(3):
    im = imconv(img_data,kernels[i])
    pos_lists = order_conv1(im)
    nodes_activ = get_nodes(pos_lists,img_data,p_inf,5)
    print(nodes_activ)
    nodes_activ = list(map(int,nodes_activ))
    color_map = []
    if len(nodes_activ) != 0:
        plt.figure(i+20)
        for node in g:
            if node in nodes_activ:
                color_map.append('green')
            else:
                color_map.append('red')
        plt.figure(i)
        nx.draw(g,node_size=50,node_color=color_map,pos=nx.spring_layout(g))
        plt.savefig('activ_node_'+str(i)+'.png')
plt.show()

#7.对应到word2vec上到图
lowdim = lowdim.tolist()
for i in range(size):
    im = imconv(img_data,kernels[i])
    pos_lists = order_conv1(im)
    nodes_activ = get_nodes(pos_lists,img_data,p_inf,5)
#    print(nodes_activ)
    if len(nodes_activ) != 0:
        plt.figure(i)
        plt.title('Active nodes: filter '+str(i))
        for idx,n in enumerate(nod):
            cor = lowdim[int(idx)]
            if n in nodes_activ:
                x = cor[0]
                y = cor[1]
                plt.scatter(x,y,c='g',s=50,edgecolor='w')
            else:
                x = cor[0]
                y = cor[1]
                plt.scatter(x,y,c='r',s=50,edgecolor='w')
        plt.savefig('w2v_act_n_'+str(i)+'.png')
plt.show()

#8.对应到word2vec上到图，画出每个激活节点之间的连接
#lowdim = lowdim.tolist()
for i in range(size):
    im = imconv(img_data,kernels[i])
    pos_lists = order_conv1(im)
    nodes_activ = get_nodes(pos_lists,img_data,p_inf,5)
    nodes_activ = list(map(int,nodes_activ))

#    print(nodes_activ)
    k = g.subgraph(nodes_activ)
    if len(nodes_activ) != 0:
        plt.figure(i+4)
        plt.title('Active nodes: filter '+str(i)+' and internal links')
        for idx,n in enumerate(nod):
            cor = lowdim[idx]
            n = int(n)
            if n in nodes_activ:
                x = cor[0]
                y = cor[1]
                plt.scatter(x,y,c='g',s=50,edgecolor='w')
                ac_nei = k.edges(n)
                if len(ac_nei) > 0:
                    for each in ac_nei:
                        ac_node = each[1]
                        indx = list(nod).index(str(ac_node))
                        cor_ = lowdim[indx]
                        com = list(zip(cor,cor_))
                        plt.plot(com[0],com[1],c='k',linestyle='-',alpha = 0.2)
            else:
                x = cor[0]
                y = cor[1]
                plt.scatter(x,y,c='r',s=50,edgecolor='w')
        plt.savefig('w2v_act_l_'+str(i)+'.png')
plt.show()
'''

#9.对应到word2vec上到图，每个节点及其所有邻居的连接
#lowdim = lowdim.tolist()
for i in range(size):
    im = imconv(img_data,kernels[i])
    pos_lists = order_conv1(im)
    nodes_activ = get_nodes(pos_lists,img_data,p_inf,5)
    if len(nodes_activ) != 0:
        plt.figure(i+8)
        plt.title('Active nodes of filter '+str(i)+' on WS network',fontsize=15)
        for idx,n in enumerate(nod):
            cor = lowdim[int(idx)]
            if n in nodes_activ:
                x = cor[0]
                y = cor[1]
                plt.scatter(x,y,c='g',s=50,edgecolor='w',zorder=2,alpha=0.5)
                nei = g.neighbors(int(n))
                for each in nei:
                    indx = list(nod).index(str(each))
                    cor_ = lowdim[indx]
                    com = list(zip(cor,cor_))
                    plt.plot(com[0],com[1],c='silver',linestyle='-',alpha = 0.5,zorder=1)
            else:
                x = cor[0]
                y = cor[1]
                plt.scatter(x,y,c='r',s=50,edgecolor='w',alpha=0.3,zorder=2)
        #plt.style.use('dark_background')
        plt.savefig('ws_w2v_act_nei_'+str(i)+'.png')
        plt.savefig('ws_w2v_act_nei_'+str(i)+'.eps')
plt.show()

'''
#10.画出子图及子图邻居节点
#lowdim = lowdim.tolist()
for i in range(size):
    im = imconv(img_data,kernels[i])
    pos_lists = order_conv1(im)
    nodes_activ = get_nodes(pos_lists,img_data,p_inf,5)
    nodes_activ = list(map(int,nodes_activ))
#    print(nodes_activ)
    if len(nodes_activ) != 0:
        neighbors = []
        for n in nodes_activ:
            neighbors.extend((g.neighbors(n)))
        plt.figure(i+12)
        plt.title('Filter '+str(i)+' : active nodes and neighbors')
        pos = nx.spring_layout(g)
        k = g.subgraph(nodes_activ)
        nx.draw(k, pos=pos,node_color='r',node_size=100)
        os = g.subgraph(neighbors)
        nx.draw(os, pos=pos, node_color='g',node_size=100)
        plt.savefig('graph_nei_'+str(i)+'.png')
plt.show()

#11.只画出激活节点的子图
#lowdim = lowdim.tolist()
for i in range(size):
    im = imconv(img_data,kernels[i])
    pos_lists = order_conv1(im)
    nodes_activ = get_nodes(pos_lists,img_data,p_inf,5)
    nodes_activ = list(map(int,nodes_activ))
#    print(nodes_activ)
    if len(nodes_activ) != 0:
        plt.figure(i+16)
        plt.title('Filter '+str(i)+' : actice nodes')
        pos = nx.spring_layout(g)
        k = g.subgraph(nodes_activ)
        nx.draw(k, pos=pos,node_color='r',node_size=100)
        plt.savefig('subgraph_'+str(i)+'.png')
plt.show()

#查看激活的节点之间的最短路径
#itertools.combinations:任意两个节点的组合
short_path = []
for i in combinations(nodes_activ,2):
    link_result = nx.bidirectional_dijkstra(g,int(i[0]),int(i[1]))
    short_path.append(link_result)

#all_path = path = nx.all_pairs_shortest_path(g)
'''

'''
for each in node_rec:
    print(g.edges(int(each)))
nx.draw_networkx(g)
plt.show()
'''

'''
#n = 0
for each in kernels:
    im = imconv(img_data,each)

    #存卷积之后的图片
#    imsave('fm1_'+str(n)+'.png',im)
#    n+=1
#    plt.figure()
#    plt.imshow(im,cmap=plt.cm.gray)
#    plt.show()
'''