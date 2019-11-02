import numpy as np
import pandas as pd
import scipy.sparse as sparse
from datetime import datetime
from pandas import Series
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def unique(old_list):
    # Minimize the use_id and item_id in the dataset
    count = 1
    dic = {}
    for i in range(len(old_list)):
        if old_list[i] in dic:
            old_list[i] = dic[old_list[i]]
        else:
            dic[old_list[i]] = count
            old_list[i] = count
            count += 1
    return old_list

def readTrainData(filename ='data/filmtrust/ratings.txt'):
    
    data = pd.read_table(filename, header=None, sep=' ')
    #
    # # read ml-1m
    print(data)
    row = list(data[0])
    column = list(data[1])
    value = list(data[2])

    
    row = unique(row)
    column = unique(column)
    

    numberOfUser = max(row) + 1
    numberOfItem = max(column) + 1
    
    #
    print(len(row))
    print(len(column))
    print(len(value))
    # # print(value)
    mtx = sparse.coo_matrix((value, (row, column)), shape=(numberOfUser, numberOfItem))
    mtx = mtx.todense()
    print(mtx)
    mtx = np.array(mtx)
    mdx = np.zeros([mtx.shape[0], mtx.shape[1]])
    
    mean_user = []
    for i in range(mtx.shape[0]):
        temp = mtx[i]
        if len(temp[temp != 0]) == 0:
            mean_user.append(0)
        else:
            mean_user.append(np.mean(temp[temp != 0]))
    for i in range(mtx.shape[0]):
       for j in range(mtx.shape[1]):
           if mtx[i][j] != 0:
               mdx[i][j] = mtx[i][j]
           else:
               mdx[i][j] = mean_user[i]

   
    return mdx, numberOfUser, numberOfItem, mtx



