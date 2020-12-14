# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 20:30:07 2016

@author: xinruyue
"""

from network_dw import *
import pickle

def processing(n):
    #随机网
    E = nx.random_graphs.erdos_renyi_graph(1000,0.2)
#    print len(nx.edges(E))
    #无标度网
    B = nx.random_graphs.barabasi_albert_graph(1000,4)
#    print len(nx.edges(B))
    #小世界网

    W = nx.random_graphs.watts_strogatz_graph(1000,8,n)
#    print len(nx.edges(W))

    g = W

    #2.deepWalk
    corp = randomWalk(g,10000)
    vec,mod,nod = word_2_vec(corp,20)

    #3.pca
    pca = PCA(n_components=2)
    lowdim = pca.fit_transform(vec)
#    print lowdim

    #4.栅格化
    img_data = img_max(lowdim,48)

    return img_data

if __name__ == "__main__":
    for n in range(1,11):
        print(n)
        n_n = n/10
        print(n_n)
        img_batch = []
        output_1 = open('WS_test_'+str(n_n)+'.pkl', 'wb')
        for i in range(600):
            i_matrix = processing(n_n)
            img_batch.append(i_matrix)
            print(i)
        pickle.dump(img_batch,output_1)
    
        output_1.close()


