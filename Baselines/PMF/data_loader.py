import scipy.sparse as sparse
import os
import pandas as pd
import numpy as np

class dataLoader():
    def __init__(self, row_num = None):
        path = "../data/filmtrust/ratings.txt"
        data = pd.read_csv(path, delimiter=" ",header = None)
        print(data.head())
        self._num_user = len(np.unique(data[0])) + 1
        self._num_movie =np.max(data[1]) + 1
        #print(self._num_movie)
        row = list(data[0])
        column = list(data[1])
        value = list(data[2])
        #print(self._num_user,self._num_movie)
        #if row_num is not None:
         #  self._data = self._data[:row_num]
        #print(self._data)
        self._data = list()
        for i in range(data[0].shape[0]):
            if i < 10:
                print(row[i],column[i],value[i])
            self._data.append([int(row[i]),int(column[i]),value[i]])
        self._data = np.array(self._data)
        self._data = self._data.astype(int)
        print(self._data[:,1])
        

    def get_matrix(self, indices):
        #for d in self._data[indices]:
        #    d[0] = int(d[0])
        #    d[1] = int(d[1])
        rating = np.full((self._num_user, self._num_movie), -1)
        for d in self._data[indices]:
            rating[d[0]][d[1]] = d[2]
        return rating

    def get_sparse(self, indices = None):
        return sparse.csr_matrix((self._data[:,2],
                                  (self._data[:,0], self._data[:,1])),
                                 shape=(self._num_user, self._num_movie))

    def get_list(self, indices = None):
        if indices == None:
            return self._data[:, 0:3]
        return self._data[indices, 0:3]


    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_movie

    def __len__(self):
        return len(self._data)

    def __getitem__(self, position):
        return self._data[position]
