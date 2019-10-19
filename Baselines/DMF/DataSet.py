# -*- Encoding:UTF-8 -*-

import numpy as np
import sys
import pandas as pd
import random
import math
class DataSet(object):
    def __init__(self, fileName):
        self.data, self.shape = self.getData(fileName)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()
        self.maxRate = 5
    def getData(self, fileName):
        
        #  if fileName == 'ml-1m':
        #     print("Loading ml-1m data set...")
        #     data = []
        #     filePath = './Data/ml-1m/ratings.dat'
        #     u = 0
        #     i = 0
        #     maxr = 0.0
        #     with open(filePath, 'r') as f:
        #         for line in f:
        #             if line:
        #                 lines = line[:-1].split("::")
        #                 user = int(lines[0])
        #                 movie = int(lines[1])
        #                 score = float(lines[2])
        #                 time = int(lines[3])
        #                 data.append((user, movie, score, time))
        #                 if user > u:
        #                     u = user
        #                 if movie > i:
        #                     i = movie
        #                 if score > maxr:
        #                     maxr = score
        #     self.maxRate = maxr
        #     print("Loading Success!\n"
        #           "Data Info:\n"
        #           "\tUser Num: {}\n"
        #           "\tItem Num: {}\n"
        #           "\tData Size: {}".format(u, i, len(data)))
        #     return data, [u, i]
        # else:
        #     print("Current data set is not support!")
        #     sys.exit()


        fileName = 'data/filmtrust/ratings.txt'
        #data = pd.read_table(fileName, header=None, sep='::')
        data = pd.read_csv(fileName,header = None, sep = ' ')
        # #
        # # # read ml-1m
        #print(data)
        row = list(data[0])
        column = list(data[1])
        value = list(data[2])

        # # movieLens ml-100k
        # # data = pd.read_csv(filename)
        # # print(data)
        # # row = list(data['userId'])
        # # column = list(data['movieId'])
        # # value = list(data['rating'])

        # # print(len(set(column)))
        # index = []
        # global find
        # for i in range(len(column)):
        #     find = False
        #     for j in range(len(index)):
        #         if column[i] == index[j]:
        #             column[i] = j
        #             find = True
        #     if not find:
        #         index.append(column[i])
        #         column[i] = len(index) - 1

        # # print(index)
        # #     # print(column)
        numberOfUser = max(row)
        numberOfItem = max(column)
        #print(numberOfUser,numberOfItem)
        #
        # # print(value)
        #mtx = sparse.coo_matrix((value, (row, column)), shape=(numberOfUser, numberOfItem))
        #mtx = mtx.todense()
        #mtx = np.array(mtx)
        #mdx = np.zeros([mtx.shape[0], mtx.shape[1]])
        #for i in range(mtx.shape[0]):
        #    for j in range(mtx.shape[1]):
        #         if mtx[i][j] != 0:
        #             mdx[i][j] = mtx[i][j]
        #         else:
        #             mdx[i][j] = np.mean(mtx[i])

        # mtx_np = mtx.tocsr()
        #
        # # 观察矩阵稀疏情况
        # plt.spy(mtx.todense())

        # 读jester-data
        #data = pd.read_excel(fileName, header=None,sep=" ")
        #numberOfUser = data.shape[0]
        #numberOfItem = data.shape[1] - 1
        #data = np.array(data)
        #data_new = [] 
        #for row in range(numberOfUser):
        #    for col in range(1, data.shape[1]):
        #         if data[row][col] != 99:
        #             data_new.append([row, col - 1, (data[row][col] + 10) / 20])
        #data_nml = Normalize(data)
        #data_nml = np.reshape(data_nml,(-1, 101))
        #print(data_new)
        #mtx_np = data_nml[:, 1:]

        data_new = []
        for i in range(len(row)):
            data_new.append((row[i] - 1, column[i] - 1, value[i] / 5 ))
        #print(data_new)
        return data_new,[numberOfUser, numberOfItem]

    def getTrainTest(self):
        # data = self.data
        # data = sorted(data, key=lambda x: (x[0], x[3]))
        # train = []
        # test = []
        # for i in range(len(data)-1):
        #     user = data[i][0]-1
        #     item = data[i][1]-1
        #     rate = data[i][2]
        #     if data[i][0] != data[i+1][0]:
        #         test.append((user, item, rate))
        #     else:
        #         train.append((user, item, rate))

        # test.append((data[-1][0]-1, data[-1][1]-1, data[-1][2]))
        # return train, test
        length = len(self.data)
        pct = 0.75
        index = round(length * pct)
        indexes = np.array(range(0, length))
        #print(index, indexes)
        random.shuffle(indexes)
        trn_idxes = indexes[0:index]
        tst_idxes = indexes[index:length]
        print(trn_idxes,tst_idxes)
        #train = self.data[trn_idxes]
        #test = self.data[tst_idxes]
        train = []
        test = []
        for i in trn_idxes:
            train.append((self.data[i][0],self.data[i][1],self.data[i][2]))
        for i in tst_idxes:
            test.append((self.data[i][0],self.data[i][1],self.data[i][2]))
        print(len(test))
        return train, test


    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            # for t in range(negNum):
            #     j = np.random.randint(self.shape[1])
            #     while (i[0], j) in self.trainDict:
            #         j = np.random.randint(self.shape[1])
            #     user.append(i[0])
            #     item.append(j)
            #     rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        for s in testData:
            tmp_user = []
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_user.append(u)
            tmp_item.append(i)
            # neglist = set()
            # neglist.add(i)
            # for t in range(negNum):
            #     j = np.random.randint(self.shape[1])
            #     while (u, j) in self.trainDict or j in neglist:
            #         j = np.random.randint(self.shape[1])
            #     neglist.add(j)
            #     tmp_user.append(u)
            #     tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)]
