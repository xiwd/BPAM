import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None


def evaluate_model(model, testRatings,K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _K
    _model = model
    _testRatings = testRatings
    _K = K
    prediction_list = []
    #rmse_list, mae_list = [], []
    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        rmse = [r[0] for r in res]
        mae = [r[1] for r in res]
        return (rmse, mae)
    # Single thread
    #print(_testRatings[2])
    label_list = [x[2] for x in _testRatings]
    for idx in range(len(_testRatings)):
        prediction = eval_one_rating(idx)
        prediction_list.append(prediction)
    rmse = RMSE(prediction_list,label_list)
    mae = MAE(prediction_list, label_list)
    return (rmse, mae)


def eval_one_rating(idx):
    # rating = _testRatings[idx]
    # hit and NDCG
    # items = _testNegatives[idx]
    # u = rating[0]
    # gtItem = rating[1]
    # items.append(gtItem)
    # # Get prediction scores
    # map_item_score = {}
    # users = np.full(len(items), u, dtype='int32')
    # predictions = _model.predict([users, np.array(items)],
    #       for i in range(len(items)):
    #         item = items[i]
    #         map_item_score[item] = predictions[i]
    #     items.pop()
    #
    #     # Evaluate top rank list
    #     ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    #     hr = getHitRatio(ranklist, gtItem)
    #     ndcg = getNDCG(ranklist, gtItem)
    #     return (hr, ndcg)                          batch_size=100, verbose=0)

    rating = _testRatings[idx]
    users = rating[0]
    items = rating[1]
    labels = rating[2]
    #users = np.full(len(items), u, dtype='int32')
    users = np.reshape(users,(1,1))
    items = np.reshape(items,(1,1))
    predictions = _model.predict([np.array(users), np.array(items)])
    #if idx < 50:
     #   print(predictions[0,0], labels)
    #rmse = RMSE(labels, predictions)
    #mae = MAE(labels, predictions)
    return predictions[0,0]

# 计算RMSE的值
def RMSE(predict_list, y_test):
    diffMat = np.array(predict_list) - np.array(y_test)
    sqdiffMat = diffMat ** 2
    length = len(sqdiffMat)
    #print(sqdiffMat.shape)
    sqdifference = sqdiffMat.sum()
    rmse = (sqdifference / length) ** 0.5
    return rmse

# 计算MSE的值
def MAE(predict_list, y_test):
    diffMat = np.array(predict_list) - np.array(y_test)
    sqdiffMat = abs(diffMat)
    length = len(sqdiffMat)
    # print(sqdiffMat.shape)
    sqdifference = sqdiffMat.sum()
    mae = sqdifference / length
    return mae



def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
