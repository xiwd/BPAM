# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import argparse
from DataSet import DataSet
import sys
import os
import heapq
import math
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument('-dataName', action='store', dest='dataName', default='ml-1m')
    parser.add_argument('-negNum', action='store', dest='negNum', default=7, type=int)
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 64])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 64])
    # parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lr', action='store', dest='lr', default=0.0001)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=10, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
    parser.add_argument('-topK', action='store', dest='topK', default=10)

    args = parser.parse_args()

    classifier = Model(args)
    start = datetime.now()
    classifier.run()
    end = datetime.now()
    print("the train time is:", end - start)


class Model:
    def __init__(self, args):
        self.dataName = args.dataName
        self.dataSet = DataSet(self.dataName)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate

        self.train = self.dataSet.train
        self.test = self.dataSet.test

        self.negNum = args.negNum
        self.testNeg = self.dataSet.getTestNeg(self.test, 99)
        self.add_embedding_matrix()

        self.add_placeholders()

        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.add_model()

        self.add_loss()

        self.lr = args.lr
        self.add_train_step()

        self.checkPoint = args.checkPoint
        self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop


    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding())
        self.item_user_embedding = tf.transpose(self.user_item_embedding)
        #print(user_item_embedding, item_user_embedding)

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)
        print(user_input)
        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            #print(user_input)
            print(user_W1)
            #user_input = np.matrix(user_input)
            #user_W1 = np.matrix(user_W1)

            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (norm_item_output* norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

    def add_loss(self):
        # regRate = self.rate / self.maxRate
        # losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        # loss = -tf.reduce_sum(losses)
        rmse = (self.rate - self.y_) ** 2
        loss = tf.reduce_sum(rmse)
        # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = loss
    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        #if os.path.exists(self.checkPoint):
         #   [os.remove(f) for f in os.listdir(self.checkPoint)]
        #else:
         #   os.mkdir(self.checkPoint)

    def run(self):
        best_rmse = 10
        best_mae = 10
        best_epoch = -1
        print("Start Training!")
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            self.run_epoch(self.sess)
            print('='*50)
            print("Start Evaluation!")
            rmse, mae = self.evaluate(self.sess, self.topK)
            print("Epoch ", epoch, "rmse: {}, mae: {}".format(rmse, mae))
            if best_rmse > rmse or best_mae > mae:
                best_rmse = rmse
                best_mae = mae
                best_epoch = epoch
                self.saver.save(self.sess, self.checkPoint)
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
        print("Best rmse: {}, mae: {}, At Epoch {}".format(best_rmse, best_mae, best_epoch))
        print("Training complete!")

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0

        def RMSE(preds, truth):
            diffmat = np.array(preds) - np.array(truth)
            #print(diffmat.shape)
            diffmat_sqre = np.square(diffmat)
            #print(diffmat_sqre.shape)
            #print(diffmat_sqre)
            length = len(diffmat_sqre)
            difference = 0
            count = 0
            for i in range(length):
                #if i < 5:
                 #    print(diffmat_sqre[i])
                 for j in range(len(diffmat_sqre[i])):
                     difference += diffmat_sqre[i][j]
                     count += 1
            #print(count)
            #print(difference.shape)
            diffmat_mean = difference / count
            rmse = diffmat_mean ** 0.5
            #print(rmse)
            return rmse
            #return np.sqrt(np.mean(np.square(preds-truth)))

        def MAE(preds, truth):
            diffmat = np.array(preds) - np.array(truth)
            diffmat_abs = abs(diffmat)
            #print(diffmat_sqre)
            length = len(diffmat_abs)
            difference = 0
            count = 0
            for i in range(length):
                #if i < 5:
                 #    print(diffmat_sqre[i])
                 for j in range(len(diffmat_abs[i])):
                     difference += diffmat_abs[i][j]
                     count += 1
            #print(difference.shape)
            mae = difference / count
            return mae
            #return np.sqrt(np.mean(np.abs(np.array(preds)-np.array(truth))))

        predict_list = []
        label_list = []
        test_u, test_i,test_r = self.dataSet.getInstances(self.test, self.negNum)
        test_len = len(self.test)
        num_batches = len(self.test) // self.batchSize + 1
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([test_len, (i+1)*self.batchSize])
            test_u_batch = test_u[min_idx: max_idx]
            test_i_batch = test_i[min_idx: max_idx]
            test_r_batch = test_r[min_idx: max_idx]

            feed_dict = self.create_feed_dict(test_u_batch, test_i_batch)
       # for i in range(len(self.test)):
            #target = self.test[i][2]
            #users = self.test[i][0]
            #items = self.test[i][1]
            #users = np.reshape(users,(1,1))
            #items = np.reshape(items,(1,1))
            #print(target, users, items)
            # predictions = _model.predict([np.array(users), np.array(items)])
            #feed_dict = self.create_feed_dict(users,items)
            #if 
            predict = sess.run(self.y_, feed_dict=feed_dict)
            predict_list.append(predict)
            label_list.append(test_r_batch)
            #print(predict[0:5], print(test_r_batch[0:5]))
        rmse = RMSE(predict_list, label_list)
        mae = MAE(predict_list, label_list)
        return rmse, mae
            

if __name__ == '__main__':
    main()
