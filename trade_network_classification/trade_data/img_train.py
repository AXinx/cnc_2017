#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:23:04 2017

@author: xinruyue
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import pickle
import numpy as np
import random

BATCH_SIZE = 100

#data
file_path = '/Users/xinruyue/xin/cnc_2017/trade_network_classification/trade_data/enriched-0-5'
print(file_path)
#prepare train data
f1_ = os.path.join(file_path,'train_data_0.pkl')
f1 = open(f1_,'rb')
content1 = pickle.load(f1,encoding='iso-8859-1')
f2_ = os.path.join(file_path,'train_data_5.pkl')
f2 = open(f2_,'rb')
content2 = pickle.load(f2,encoding='iso-8859-1')
d_train_data = content1 + content2

s1 = len(content1)
s2 = len(content2)
s = s1 + s2

d_train_labels = [0] * s1
d_labels_0 = [1] * s2
d_train_labels.extend(d_labels_0)

data_label_pair = list(zip(d_train_data, d_train_labels))
random.shuffle(data_label_pair)

train_data_t = list(zip(*data_label_pair))[0]
train_labels_t = list(zip(*data_label_pair))[1]

train_data = np.array(train_data_t).reshape((s,1,48,48))
train_labels = train_labels_t

train_size = train_data.shape[0]

train_data = torch.from_numpy(train_data)
train_labels = torch.LongTensor(train_labels)

#prepare val data
f3_ = os.path.join(file_path,'val_data_0.pkl')
f3 = open(f3_,'rb')
content3 = pickle.load(f3,encoding='iso-8859-1')
f4_ = os.path.join(file_path,'val_data_5.pkl')
f4 = open(f4_,'rb')
content4 = pickle.load(f4,encoding='iso-8859-1')
d_val_data = content3 + content4

s3 = len(content3)
s4 = len(content4)
s_ = s3 + s4

d_val_labels = [0] * s3
d_labels_1 = [1] * s4
d_val_labels.extend(d_labels_1)

val_data_label_pair = list(zip(d_val_data, d_val_labels))
random.shuffle(val_data_label_pair)

val_data_t = list(zip(*val_data_label_pair))[0]
val_labels_t = list(zip(*val_data_label_pair))[1]

val_data = np.array(val_data_t).reshape((s_,1,48,48))
val_labels = val_labels_t

val_data = torch.from_numpy(val_data)
val_labels = torch.LongTensor(val_labels)

#prepare test data
f5_ = os.path.join(file_path,'test_data_0.pkl')
f5 = open(f5_,'rb')
content5 = pickle.load(f5,encoding='iso-8859-1')
f6_ = os.path.join(file_path,'test_data_5.pkl')
f6 = open(f6_,'rb')
content6 = pickle.load(f6,encoding='iso-8859-1')
d_test_data = content5 + content6

s5 = len(content3)
s6 = len(content4)
s_ = s5 + s6

d_test_labels = [0] * s5
d_labels_2 = [1] * s6
d_test_labels.extend(d_labels_2)

test_data_label_pair = list(zip(d_test_data, d_test_labels))
random.shuffle(test_data_label_pair)

test_data_t = list(zip(*test_data_label_pair))[0]
test_labels_t = list(zip(*test_data_label_pair))[1]

test_data = np.array(test_data_t).reshape((s_,1,48,48))
test_labels = test_labels_t

test_data = torch.from_numpy(test_data)
test_labels = torch.LongTensor(test_labels)

#model
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,15,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(15,30,5)
        self.fc1 = nn.Linear(30*9*9,300)
        self.fc2 = nn.Linear(300,2)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,30*9*9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

net = Net()
net.double()

#代价函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

full_mean_los = []
ep = []
acu = []
acu_train = []
for epoch in range(30):
    los = []
    steps = int(train_size / BATCH_SIZE)
    running_loss = 0.0
    acu_t = []
    for step in range(steps):
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)

        inputs = train_data[offset:(offset + BATCH_SIZE), :, :, :]
        labels = train_labels[offset:(offset + BATCH_SIZE)]
        inputs,labels = Variable(inputs),Variable(labels)
        #train
        optimizer.zero_grad()
        train_outputs = net(inputs)
        _,pred = torch.max(train_outputs.data, 1)
        tol = labels.size(0)
        corr = (pred == labels.data[0]).sum()
        accu = corr/tol
        acu_t.append(accu)
        loss = criterion(train_outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        los.append(running_loss)
        #print('[%d, %5d] loss: %3f'%(epoch+1, step, running_loss))
        running_loss = 0.0
    acu_t_e = np.mean(acu_t)
    acu_train.append(acu_t_e)
    print('One epoch training finished')
    mean_loss = np.mean(los)
    full_mean_los.append(mean_loss)
    ep.append(epoch)
    correct = 0
    total = 0
    val_outputs = net(Variable(val_data))
    _,predicted = torch.max(val_outputs.data, 1)
    total = val_labels.size(0)
    correct = (predicted == val_labels).sum()
    accuracy = correct/total
    acu.append(accuracy)
    print('Accuracy of the network on the test images: %d %%'% (100*accuracy))

test_outputs = net(Variable(test_data))
_,test_predicted = torch.max(test_outputs.data, 1)
test_total = test_labels.size(0)
test_correct = (test_predicted == test_labels).sum()
test_accuracy = test_correct/test_total
print('Accuracy of the network on the test images: %d %%'% (100*test_accuracy))



with open('epoch.txt','w') as f1:
    for each in ep:
        f1.write(str(each))
        f1.write('\n')

with open('loss.txt','w') as f2:
    for each in full_mean_los:
        f2.write(str(each))
        f2.write('\n')

with open('val_acu.txt','w') as f3:
    for each in acu:
        f3.write(str(each))
        f3.write('\n')
        
with open('train_acu.txt','w') as f4:
    for each in acu_train:
        f4.write(str(each))
        f4.write('\n')