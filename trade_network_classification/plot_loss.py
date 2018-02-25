#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:45:58 2017

@author: xinruyue
"""
import matplotlib.pyplot as plt

f1 = open('epoch.txt', 'r')
f2 = open('val_acu.txt', 'r')
f3 = open('loss.txt', 'r')

ep = []
for each in f1:
    each = each.strip('\n')
    ep.append(each)

acu = []
for each in f2:
    each = each.strip('\n')
    each = float(each)
    acu.append(each)

los = []
for each in f3:
    each = each.strip('\n')
    los.append(each)
    
err = list(map(lambda x: 1-x, acu))

'''
s = 12
fig = plt.figure()
plt.title('Loss and accuracy', fontsize = s)
ax1 = fig.add_subplot(1,1,1)
ax1.plot(ep, los, 'r', label='loss')
plt.legend(bbox_to_anchor=(1.0,0.15))
ax1.set_ylabel('Loss',fontsize = s)
ax2 = ax1.twinx()
ax2.plot(ep, acu, 'g', label='accuracy')
plt.legend(bbox_to_anchor=(1.0,0.95))
ax2.set_ylabel('Accuracy',fontsize = s)
ax1.set_xlabel('Epoch',fontsize = s)
plt.show()
plt.savefig('loss_acu.png')
'''
s1 = 15
s2 = 12
fig = plt.figure()
plt.title('Loss and Error', fontsize = s1)
ax1 = fig.add_subplot(1,1,1)
l1 = ax1.plot(ep, los, 'r', label='loss')
#plt.legend(bbox_to_anchor=(1.0,0.15))
ax1.set_ylabel('Loss',fontsize = s2)
ax2 = ax1.twinx()
l2 = ax2.plot(ep, err, 'g', label='error')
ls = l1+l2
labs = [l.get_label() for l in ls]
ax1.legend(ls, labs, bbox_to_anchor=(1.0,0.95))
ax2.set_ylabel('Error rate',fontsize = s2)
ax1.set_xlabel('Epoch',fontsize = s2)
plt.savefig('loss_err.png')
plt.savefig('loss_err.eps')
plt.show()
