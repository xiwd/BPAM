import numpy as np
import tensorflow as tf
from keras import initializers
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dot, Lambda, multiply, Reshape, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras import backend as K
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import argparse
import DMF_implicit2
import MLP2

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepF.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[512,256,128,64]',
                        help="MLP layers. Note that the first layer is the concatenation "
                             "of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--userlayers', nargs='?', default='[512, 64]',
                        help="Size of each user layer")
    parser.add_argument('--itemlayers', nargs='?', default='[1024, 64]',
                        help="Size of each item layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='sgd',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    parser.add_argument('--dmf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for DMF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


def get_model(train, num_users, num_items, userlayers, itemlayers, layers):
    # assert len(layers) == len(reg_layers)
    dmf_num_layer = len(userlayers) #Number of layers in the DMF
    mlp_num_layer = len(layers) #Number of layers in the MLP
    user_matrix = K.constant(getTrainMatrix(train))
    item_matrix = K.constant(getTrainMatrix(train).T)
    #user_matrix = getTrainMatrix(train)
    #item_matrix = user_matrix.T
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    
    # Embedding layer
    user_rating= Lambda(lambda x: tf.gather(user_matrix, tf.to_int32(x)))(user_input)
    item_rating = Lambda(lambda x: tf.gather(item_matrix, tf.to_int32(x)))(item_input)
    user_rating = Reshape((num_items, ))(user_rating)
    item_rating = Reshape((num_users, ))(item_rating)

    # DMF part
    userlayer = Dense(userlayers[0],  activation="linear" , name='user_layer0')
    itemlayer = Dense(itemlayers[0], activation="linear" , name='item_layer0')
    dmf_user_latent = userlayer(user_rating)
    dmf_item_latent = itemlayer(item_rating)
    for idx in range(1, dmf_num_layer):
        userlayer = Dense(userlayers[idx],  activation='relu', name='user_layer%d' % idx)
        itemlayer = Dense(itemlayers[idx],  activation='relu', name='item_layer%d' % idx)
        dmf_user_latent = userlayer(dmf_user_latent)
        dmf_item_latent = itemlayer(dmf_item_latent)
    dmf_vector = multiply([dmf_user_latent, dmf_item_latent])

    # MLP part 
    MLP_Embedding_User = Dense(layers[0]//2, activation="linear" , name='user_embedding')
    MLP_Embedding_Item  = Dense(layers[0]//2, activation="linear" , name='item_embedding')
    mlp_user_latent = MLP_Embedding_User(user_rating)
    mlp_item_latent = MLP_Embedding_Item(item_rating)
    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    for idx in range(1, mlp_num_layer):
        layer = Dense(layers[idx], activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate DMF and MLP parts
    predict_vector = concatenate([dmf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_normal(),
                       name="prediction")(predict_vector)
    
    model_ = Model(inputs=[user_input, item_input],
                   outputs=prediction)
    
    return model_

def getTrainMatrix(train):
    num_users, num_items = train.shape
    train_matrix = np.zeros([num_users, num_items], dtype=np.float)
    for (u, i) in train.keys():
        train_matrix[u][i] = train[u,i]
    return train_matrix

def load_pretrain_model1(model, dmf_model, dmf_layers,):
    # get_layer返回的是list
    # MF embeddings
    dmf_user_embeddings = dmf_model.get_layer('user_layer0').get_weights()
    dmf_item_embeddings = dmf_model.get_layer('item_layer0').get_weights()
    model.get_layer('user_layer0').set_weights(dmf_user_embeddings)
    model.get_layer('item_layer0').set_weights(dmf_item_embeddings)
        
    # DMF layers
    for i in range(1, len(dmf_layers)):
        dmf_user_layer_weights = dmf_model.get_layer('user_layer%d' % i).get_weights()
        model.get_layer('user_layer%d' % i).set_weights(dmf_user_layer_weights)
        dmf_item_layer_weights = dmf_model.get_layer('item_layer%d' % i).get_weights()
        model.get_layer('item_layer%d' % i).set_weights(dmf_item_layer_weights)
        
    # Prediction weights
    dmf_prediction = dmf_model.get_layer('prediction').get_weights()
    # print(dmf_prediction[0]) # 0是权重，是一个列表，每一个权重都单独再用一个列表包住，形如：[[1.1], [3.4], [1.2], [2.5]]
    # print(dmf_prediction[1]) # 1是偏置，也是一个列表，形如：[1.4]
    new_weights = np.concatenate((dmf_prediction[0], np.array([[0,]] * dmf_layers[-1])), axis=0)
    new_b = dmf_prediction[1]
    model.get_layer('prediction').set_weights([new_weights, new_b]) 
    return model

def load_pretrain_model2(model, mlp_model, mlp_layers):
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('user_embedding').set_weights(mlp_user_embeddings)
    model.get_layer('item_embedding').set_weights(mlp_item_embeddings)
        
    # MLP layers
    for i in range(1, len(mlp_layers)):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)
    
    # Prediction weights
    dmf_prediction = model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((dmf_prediction[0][:mlp_layers[-1]], mlp_prediction[0]), axis=0)
    new_b = dmf_prediction[1] + mlp_prediction[1]
    # 0.5 means the contributions of MF and MLP are equal
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b]) 
    return model

def get_train_instances(train):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    # for (u, i) in train.keys():
    #     # positive instance
    #     user_input.append(u)
    #     item_input.append(i)
    #     labels.append(1)
    #     # negative instances
    #     for t in range(num_negatives):
    #         j = np.random.randint(num_items)
    #         # while train.has_key((u, j)):
    #         while (u, j) in train.keys():
    #             j = np.random.randint(num_items)
    #         user_input.append(u)
    #         item_input.append(j)
    #         labels.append(0)


    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(train[u,i])

    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    userlayers = eval(args.userlayers)
    itemlayers = eval(args.itemlayers)
    layers = eval(args.layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.epochs
    verbose = args.verbose
    dmf_pretrain = args.dmf_pretrain
    mlp_pretrain = args.mlp_pretrain
            
    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("DeepCF arguments: %s " % args)
    model_out_file = 'Pretrain/%s_DeepCF_%s_%s_%s_%d.h5' %(args.dataset, args.userlayers, args.itemlayers, args.layers, time())

    # Loading data
    t1 = time()
    start = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings = dataset.trainMatrix, dataset.testRatings   
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(train, num_users, num_items, userlayers, itemlayers, layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='mean_squared_error')  # change mean_squared_error from binary_crossentropy
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='mean_squared_error')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='mean_squared_error')
    
    # Load pretrain model
    if dmf_pretrain != '' and mlp_pretrain != '':
        dmf_model = DMF_implicit2.get_model(train, num_users, num_items, userlayers, itemlayers)
        dmf_model.load_weights(dmf_pretrain)
        model = load_pretrain_model1(model, dmf_model,  userlayers)
        del dmf_model
        mlp_model = MLP2.get_model(train, num_users, num_items, layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model2(model, mlp_model,  layers)
        del mlp_model
        print("Load pretrained DMF (%s) and MLP (%s) models done. " % (dmf_pretrain, mlp_pretrain))
        
    # Check Init performance
    (rmse, mae) = evaluate_model(model, testRatings, topK, evaluation_threads)
    #rmse, mae = np.array(rmses).mean(), np.array(maes).mean()
    #rmse = rmse ** 0.5
    print('Init: rmse = %.4f, mae = %.4f' % (rmse, mae))
    best_rmse, best_mae, best_iter = rmse, mae, -1
    # 初始化后先保存一次，万一用预训练模型初始化的是最好的，后面性能一直变差，那就算直接停止程序，最好的也保存下来了
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True) 
        
    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch % verbose == 0:
            (rmses, maes) = evaluate_model(model, testRatings, topK, evaluation_threads)
            rmse, mae, loss = rmses, maes, hist.history['loss'][0]
            print('Iteration %d [%.1f s]: rmse = %.4f, mae = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, rmse, mae, loss, time()-t2))
            if rmse < best_rmse:
                best_rmse, best_mae, best_iter = rmse, mae, epoch
                if args.out > 0: # 每次有更好的结果时就保存模型
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  rmse = %.4f, mae = %.4f. " %(best_iter, best_rmse, best_mae))
    t = time()
    print("the time is: ", t - start)
    if args.out > 0:
        print("The best DeepCF model is saved to %s" % model_out_file)
