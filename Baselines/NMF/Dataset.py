'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + "_train.csv")
        self.testRatings = self.load_rating_file_as_list(path + "_test.csv")

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        # ratingList = []
        # with open(filename, "r") as f:
        #     line = f.readline()
        #     # while line != None and line != "":
        #     while line is not None and line != "":
        #         arr = line.split("\t")
        #         user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
        #         ratingList.append([user, item, rating / 5])
        #         line = f.readline()
        # return ratingList
        #data = pd.read_table(filename, header=None, sep='::')
        #
        # # read ml-1m
        data = pd.read_csv(filename, header = None)
        ratingList = []
        print(data.head())
        #print('ok!')
        row = list(data[0])
        column = list(data[1])
        value = list(data[2]/5)
        for i in range(len(row)):
            ratingList.append([row[i], column[i], value[i]])
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        # num_users, num_items = 0, 0
        # with open(filename, "r") as f:
        #     line = f.readline()
        #     # while line != None and line != "":
        #     while line is not None and line != "":
        #         arr = line.split("\t")
        #         u, i = int(arr[0]), int(arr[1])
        #         num_users = max(num_users, u)
        #         num_items = max(num_items, i)
        #         line = f.readline()
        # # Construct matrix
        # mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        # with open(filename, "r") as f:
        #     line = f.readline()
        #     # while line != None and line != "":
        #     while line is not None and line != "":
        #         arr = line.split("\t")
        #         user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])

        #         mat[user, item] = rating / 5
        #         line = f.readline()
        # return mat
        data = pd.read_csv(filename, header = None)
        print(data.head())
        #print('ok!!!!!!!!!!!!!')
        row = list(data[0])
        column = list(data[1])
        value = list(data[2]/5)
        numberOfUser = int(max(row) + 1)
        numberOfItem = int(max(column) + 1)
        print(numberOfUser,numberOfItem)
        mat = sp.dok_matrix((numberOfUser, numberOfItem),dtype=np.float32)
        #mtx = sp.coo_matrix((value, (row, column)), shape=(numberOfUser, numberOfItem))
        #mdx = mtx.todense()
        for i in range(len(row)):
            mat[row[i],column[i]] = value[i]
        return mat
        #return data















# '''
# Created on Aug 8, 2016
# Processing datasets. 

# @author: Xiangnan He (xiangnanhe@gmail.com)
# '''
# import scipy.sparse as sp
# import numpy as np

# class Dataset(object):
#     '''
#     classdocs
#     '''

#     def __init__(self, path):
#         '''
#         Constructor
#         '''
#         self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
#         self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
#         self.testNegatives = self.load_negative_file(path + ".test.negative")
#         assert len(self.testRatings) == len(self.testNegatives)
        
#         self.num_users, self.num_items = self.trainMatrix.shape
        
#     def load_rating_file_as_list(self, filename):
#         ratingList = []
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line != None and line != "":
#                 arr = line.split("\t")
#                 user, item = int(arr[0]), int(arr[1])
#                 ratingList.append([user, item])
#                 line = f.readline()
#         return ratingList
    
#     def load_negative_file(self, filename):
#         negativeList = []
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line != None and line != "":
#                 arr = line.split("\t")
#                 negatives = []
#                 for x in arr[1: ]:
#                     negatives.append(int(x))
#                 negativeList.append(negatives)
#                 line = f.readline()
#         return negativeList
    
#     def load_rating_file_as_matrix(self, filename):
#         '''
#         Read .rating file and Return dok matrix.
#         The first line of .rating file is: num_users\t num_items
#         '''
#         # Get number of users and items
#         num_users, num_items = 0, 0
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line != None and line != "":
#                 arr = line.split("\t")
#                 u, i = int(arr[0]), int(arr[1])
#                 num_users = max(num_users, u)
#                 num_items = max(num_items, i)
#                 line = f.readline()
#         # Construct matrix
#         mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line != None and line != "":
#                 arr = line.split("\t")
#                 user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
#                 if (rating > 0):
#                     mat[user, item] = 1.0
#                 line = f.readline()    
#         return mat
