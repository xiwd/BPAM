import numpy as np
import math
import  evaluatingIndicator
from sklearn.preprocessing import MinMaxScaler
import  BPnetworkAttention
import  random
from pandas import DataFrame
from datetime import datetime
import pandas as pd


def rand(a, b):
    # Generate random values between a and b
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    # Generate a matrix with m*n dimensions
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def general_attention(number, hidden_n):
    # Generate the weight of atttntion
    # number：the number of users' neighbors
    # hidden_n: the number of neurons in the hidden layers
    att_weights = make_matrix(number, hidden_n)
    att_correct = make_matrix(number, hidden_n)


    return att_weights, att_correct


def data_spilit(x, y, pct):
    # x：input data
    # y：the label
    # pct：the percent of training data
    length = len(y)
    index = round(length * pct)
    indexes = np.array(range(0, length))
    random.shuffle(indexes)
    trn_idxes = indexes[0:index]
    tst_idxes = indexes[index:length]
    # print(trn_idxes)
    # print(x[trn_idxes,:])
    x_train = x[trn_idxes, :]
    x_test = x[tst_idxes, :]
    y_train = y[trn_idxes, :]
    y_test = y[tst_idxes, :]
    return x_train, x_test, y_train, y_test, indexes

def number_of_zero(data):
    count = 0
    for e in data:
        if e == 0:
            count += 1
    return count

def predict_item_and_user(number_of_x, number_of_y, neighbor, mtx_np, k, att_weights, att_correct, input_n, hidden_n, mtx_ds,ar):
    rmse_sum_difference = 0
    mae_sum_difference = 0
    count_sum = 0
    RMSE_list = list()
    MAE_list = list()

    start = datetime.now()
    for id in range(1, number_of_x):  
        id_of_neighbors = list(neighbor[id][:])

        # Import rating information of all neighbors of users into X_np
        X = []
        att_weights_knn = []
        att_correct_knn = []
        #print ("id %d:" % id, id_of_neighbors)
        for neighbor_id in id_of_neighbors:
            X.append(mtx_ds[neighbor_id])
            att_weights_knn.append(att_weights[neighbor_id])
            att_correct_knn.append(att_correct[neighbor_id])
        X_np = np.array(X, dtype=float)
        X_np = np.reshape(X_np, (k, number_of_y))

        # Store user rating information
        y = mtx_np[id]
        y = np.reshape(y, (1, number_of_y))

        # Transpose, each line denotes the rating information of all neighbors
        X_np = X_np.T
        y = y.T

        # Pick out items that have been evaluated by user u
        y_new = []
        x_new = []
        origine_index = []  # Record the original index
        count = 0
        # print(y)
        for keys in range(number_of_y):
            if y[keys] >  0:
                x_new.append(X_np[keys])
                y_new.extend(y[keys])
                origine_index.append(keys)
                count += 1
        # Convert list to array form for easy training
        y_new = np.reshape(y_new, (count, 1))
        x_new = np.reshape(x_new, (count, k))
  

        # Split data into training and test sets
        X_train, X_test, y_train, y_test, indexs = data_spilit(x_new, y_new, 0.75)
 

        # Creating a neural network
        nm = BPnetworkAttention.BPNeuralNetwork()
        nm.setup(input_n, hidden_n, 1)
        attention_rate = ar  # attention ratio
        att_weights_knn, att_correct_knn = nm.train(X_train, y_train, att_weights_knn, att_correct_knn,
                                                     attention_rate, 100, 0.0001, 0.1)  # Importing data into BP neural network for training


        #Update att_weight and att_correct
        i = 0
        for neighbor_id in id_of_neighbors:
            att_weights[neighbor_id] = att_weights_knn[i]
            att_correct[neighbor_id] = att_correct_knn[i]
            i += 1

        # Predict the test set and get the test results
        #print(att_weights_knn)
        predict_list = []  # for storing the predicted ratings
        for i in range(len(X_test)):
            predict = nm.predict(X_test[i], att_weights_knn, attention_rate)
            predict_list.append(predict)            


        predict_list = np.array(predict_list)


        RMSEsqdifference, length ,RMSE = evaluatingIndicator.RMSE(predict_list, y_test, id)  # return RMSE
        rmse_sum_difference += RMSEsqdifference
        count_sum += length

        MAEsqdifference, length, MAE = evaluatingIndicator.MAE(predict_list, y_test, id)  # return MAE
        mae_sum_difference += MAEsqdifference
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)

    end = datetime.now()


    RMSE = (rmse_sum_difference / count_sum) ** 0.5
    print('the RMSE about the prediction of all users: %f' % (RMSE * 5))

    MAE = mae_sum_difference / count_sum
    print('the MAE about the prediction of all  users: %f' % (MAE * 5))
    RMSE_list.append(RMSE)
    MAE_list.append(MAE)

    return  RMSE, att_weights, att_correct, RMSE_list, MAE_list



def rating_predict(numberOfUser, numberOfItem, neighbor_user,  mtx_np, k_user, mtx_ds,ar):
    '''
    :param numberOfUser: the number of users
    :param numberOfItem: the number of items
    :param neighbor_user: the KNN neighbor matrix of users
    :param mtx_np: rating matrix
    :param k: the number of users' neighbors
    :param mtx_ds: Rating matrix after data preprocessing
    :ar: Attention rate
    :return:
    '''


    mm_mtx_np = np.array(mtx_np) / 5
    mm_mtx_np_T = mm_mtx_np.T
    mtx_ds = mtx_ds / 5

    input_n = k_user  # the number of neurons in input layer
    hidden_n = int(k_user / 2)  # the number of neurons in hidden layer
    att_weights_user, att_correct_user = general_attention(numberOfUser, hidden_n)
    
    print(mm_mtx_np)
    print(mtx_ds)
    RMSE_user, att_weights_user, att_correct_user, RMSE_list, MAE_list = predict_item_and_user(
        numberOfUser, numberOfItem, neighbor_user, mm_mtx_np, k_user,
        att_weights_user, att_correct_user, input_n, hidden_n, mtx_ds,ar)




