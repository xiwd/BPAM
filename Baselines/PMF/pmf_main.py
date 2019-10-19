from __future__ import print_function
from evaluations import *
from pmf_model import *
from data_loader import dataLoader
import random

def data_spilit(x, pct = 0.8):
    length = len(x)
    index = int(length * pct)

    indexes = np.array(range(0, length))
    random.shuffle(indexes)
    trn_idxes = indexes[0:index]
    tst_idxes = indexes[index:length]

    x_train = x[trn_idxes, :]
    x_test = x[tst_idxes, :]
    return x_train, x_test


print('PMF Recommendation Model Example')

# choose dataset name and load dataset, 'ml-1m', 'ml-10m'
dataloader = dataLoader()


# set split ratio
ratio = 0.75
train_data, test_and_vali_data = data_spilit(dataloader._data, 0.75)
vali_data, test_data = data_spilit(test_and_vali_data, 0.5)

NUM_USERS = dataloader.num_user
NUM_ITEMS = dataloader.num_item
print('dataset density:{:f}'.format(len(dataloader._data)*1.0/(NUM_USERS*NUM_ITEMS)))


R = np.zeros([NUM_USERS, NUM_ITEMS])
for ele in train_data:
    R[int(ele[0]), int(ele[1])] = float(ele[2])

# construct model
print('training model.......')
lambda_alpha = 0.01
lambda_beta = 0.01
latent_size = 20
lr = 3e-5
iters = 500
model = PMF(R=R, lambda_alpha=lambda_alpha, lambda_beta=lambda_beta, latent_size=latent_size, momuntum=0.9, lr=lr, iters=iters, seed=1)
print('parameters are:ratio={:f}, reg_u={:f}, reg_v={:f}, latent_size={:d}, lr={:f}, iters={:d}'.format(ratio, lambda_alpha, lambda_beta, latent_size,lr, iters))
U, V, train_loss_list, vali_rmse_list = model.train(train_data=train_data, vali_data=vali_data)

print('testing model.......')
preds = model.predict(data=test_data)
test_rmse = RMSE(preds, test_data[:, 2])

print('test rmse:{:f}'.format(test_rmse))
