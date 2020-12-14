# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 20:30:07 2016

@author: xinruyue
"""

from network_dw import *
import pickle
import multiprocessing
import datetime

def processing():
    #随机网
    E = nx.random_graphs.erdos_renyi_graph(1000,0.2)
#    print len(nx.edges(E))
    #无标度网
    B = nx.random_graphs.barabasi_albert_graph(1000,4)
#    print len(nx.edges(B))
    #小世界网
    W = nx.random_graphs.watts_strogatz_graph(1000,16,0.1)
#    print len(nx.edges(W))

    g = W

    #2.deepWalk
    corp = randomWalk(g,10000)
    vec,mod,nod = word_2_vec(corp,20)

    #3.pca
    pca = PCA(n_components=2)
    lowdim = pca.fit_transform(vec)

    #4.栅格化
    img_data = img_max(lowdim,48)

    return img_data

def subprocessing(lis):
    print(str(datetime.datetime.now()) + "\tPID = " + str(multiprocessing.current_process().pid) + " start...")
    img_batch = []
    for i in range(lis):
        i_matrix = processing()
        img_batch.append(i_matrix)
    print(str(datetime.datetime.now()) + "\tPID = " + str(multiprocessing.current_process().pid) + " finished.")
    return img_batch


if __name__ == "__main__":
    process = multiprocessing.cpu_count() - 2
    print(process)
    pool = multiprocessing.Pool(processes=process)
    result = []
    for i in range(5):
        result.append(pool.apply_async(subprocessing, (1000,)))
    pool.close()
    pool.join()

    img_batch = []
    for res in result:
        r = res.get()
        for i in r:
            img_batch.append(i)
    output_1 = open('WS_train_16.pkl', 'wb')
    pickle.dump(img_batch, output_1)
    print('done')
    output_1.close()