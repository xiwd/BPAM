import numpy as np


# Calculate the value of RMSE
def RMSE(predict_list, y_test,user):
    diffMat = np.array(predict_list) - np.array(y_test)
    sqdiffMat = diffMat ** 2
    length = len(sqdiffMat)
    sqdifference = sqdiffMat.sum()
    singe_user_RMSE = (sqdifference / len(sqdiffMat) ) ** 0.5

    return sqdifference, length, singe_user_RMSE * 5

# Calculate the value of MAE
def MAE(predict_list, y_test, user):
    diffMat = np.array(predict_list) - np.array(y_test)
    sqdiffMat = abs(diffMat)
    length = len(sqdiffMat)
    sqdifference = sqdiffMat.sum()
    singe_user_MAE = sqdifference / len(sqdiffMat)

    return sqdifference, length,singe_user_MAE * 5

