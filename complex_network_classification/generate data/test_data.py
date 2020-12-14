#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:35:59 2017

@author: xinruyue
"""

import pickle

f1 = open('./generate data/WS_test_2.pkl','rb')
content1 = pickle.load(f1,encoding='iso-8859-1')

print(len(content1))
