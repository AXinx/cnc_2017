#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:27:12 2017

@author: xinruyue
"""

import pandas as pd
import numpy as np
import xlrd
import pickle
import os

def get_country():
    f = open('country.txt','r')
    country = []
    for line in f:
        line = line.strip('\n')
        country.append(line)
    return country

#get F matrix
def get_f(df,country):
    size = len(country)
    f_matrix = np.zeros((size,size))
    for index,row in df.iterrows():
        imp = row[2]
        exp = row[4]
        value = row[8]
        i = country.index(imp)
        j = country.index(exp)
        f_matrix[i][j] = value
    return f_matrix

def processing(file1,y):
    # get all data
    df = pd.DataFrame()
    book = xlrd.open_workbook(file1)
    for sheet in book.sheets():
        f = pd.read_excel(file1, sheetname=sheet.name)
        df = pd.concat([df, f], ignore_index=True)

    # remove 'world'
    ex_list1 = list(df.Importer)
    ex_list2 = list(df.Exporter)
    ex_list1 = list(filter(lambda i: i != 'World', ex_list1))
    ex_list2 = list(filter(lambda i: i != 'World', ex_list2))
    df = df[df.Importer.isin(ex_list1)]
    df = df[df.Exporter.isin(ex_list2)]

    '''
    # get country
    country = df['Importer'].append(df['Exporter'])
    country = country.drop_duplicates()
    country = country.sort_values()
    country = list(country)
    '''
    
    country = get_country()

    #get each products' Sitc4
    sitc = list(df.Sitc4)
    sitc_dic = {}
    for i in range(10):
        i = str(i)
        s_code = []
        for each in sitc:
            each = str(each)
#            each = each.encode('utf-8')
#            print(type(each))
            if len(each) != 4:
                each = '%04d'%(int(each))
            if each[0] == i:
                if each not in s_code:
                    s_code.append(each)
        sitc_dic[i] = s_code

    for key,value in sitc_dic.items():
        for e_val in value:
            val = []
            val.append(e_val)
            pro_df = df[df.Sitc4.isin(val)]
            f_mat = get_f(pro_df,country)
            fil = open(y+'_'+str(e_val)+'.pkl','wb')
            pickle.dump(f_mat,fil)
            fil.close()
            print(e_val)

if __name__ == '__main__':
    file_path = './data'
    files = os.listdir(file_path)
    for each in files:
        year = ''.join(list(filter(str.isdigit,each)))
        f = os.path.join(file_path,each)
        processing(f,year)
        print(year)




