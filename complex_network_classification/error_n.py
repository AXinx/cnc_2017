#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:51:33 2017

@author: xinruyue
"""
import tensorflow as tf
import numpy as np
import pickle
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.cm as cmx
import matplotlib.colors as colors

BATCH_SIZE = 100
data_path = './data'

def data_processing(i):

    file1 = os.path.join(data_path,'BA_test_'+str(i*100)+'.pkl')
    file2 = os.path.join(data_path,'WS_test_'+str(i*100)+'.pkl')

    # prepare test datas and labels
    f3 = open(file1,'rb')
    content3 = pickle.load(f3,encoding='iso-8859-1')
    f4 = open(file2,'rb')
    content4 = pickle.load(f4,encoding='iso-8859-1')
    dummy_test_data = content3 + content4

    dummy_test_labels = np.zeros((1200,2))
    dummy_test_labels[:600, 0 ] = 1
    dummy_test_labels[600:, 1 ] = 1

    test_data_label_pair = list(zip(dummy_test_data, dummy_test_labels))
    random.shuffle(test_data_label_pair)

    test_data_temp = list(zip(*test_data_label_pair))[0]
    test_labels_temp = list(zip(*test_data_label_pair))[1]

    test_data = np.array(test_data_temp).reshape((1200,48,48,1)).astype(np.float32)
    test_labels = np.array(test_labels_temp)

    test_size = test_labels.shape[0]

    sess=tf.Session()

    #load meta graph
    saver = tf.train.import_meta_graph('model_1-79.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]

    steps = int(test_size / BATCH_SIZE)
    pre_res = []
    for step in range(steps):
        offset = (step * BATCH_SIZE) % (test_size - BATCH_SIZE)
        feed_dict = {x : test_data[offset:(offset + BATCH_SIZE)]}
        res = sess.run(y, feed_dict = feed_dict)
        res = list(np.argmax(res,1))
        pre_res += res
    a=np.array(pre_res)
    b=np.array(np.argmax(test_labels,1))
    correct = np.sum(a==b)
    total = len(pre_res)
    error = 1.0 - (1 * float(correct) / float(total))
    return error,b,a

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def plot_roc():
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for n in range(5,16):
        _,y_label,y_score = data_processing(n)
        fpr[n], tpr[n], _ = roc_curve(y_label, y_score)
        roc_auc[n] = auc(fpr[n], tpr[n])
    lw = 2
    plt.figure(figsize=(5,5))
    cmap = get_cmap(16)
    print('here')
    for i in range(5,16):
        col = cmap(i)
        plt.plot(fpr[i], tpr[i], color=col, lw=lw,
             label='ROC curve of n = {0} (area = {1:0.2f})'''.format(i*100, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC to different m in BA')
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", prop={'size':10})
    plt.savefig('wb_roc')
    plt.show()


def plot_error():
    x = []
    y = []
    y_std = []
    for n in range(11,16):
        err = []
        for i in range(10):
            error,_,_ = data_processing(n)
            err.append(error)
        ave_err = np.mean(err)
        print(ave_err)
        std = np.std(err)
        y_std.append(std)
        y.append(ave_err)
        x.append(n)
        print(x)
        print(y)
        print(y_std)
    return x,y,y_std

'''
    plt.figure()
    plt.title('test error')
    plt.xlabel('n*100')
    plt.ylabel('error')
    plt.scatter(x,y,s=5)
    plt.errorbar(x,y,yerr=y_std)
    plt.plot(x,y)
    plt.show()
    plt.savefig('error_n_1.png')
'''

if __name__ == '__main__':
    x,y,y_std = plot_error()
    with open('n_error_1200_1.txt','w') as f:
        for each in x:
            f.write(str(each))
            f.write(',')
        f.write('\n')
        for each in y:
            f.write(str(each))
            f.write(',')
        f.write('\n')
        for each in y_std:
            f.write(str(each))
            f.write(',')
        f.write('\n')
    f.close()

