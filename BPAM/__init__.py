import  GetKNearestNeighbor
import read_Data
import RatingPredict


if __name__ == "__main__":

    k_user = 10 
    ar = 2
    filename = 'data/filmtrust/ratings.txt'
    mtx_ds, numberOfUser, numberOfItem, mtx_np = read_Data.readTrainData(filename)
    neighbor_user= GetKNearestNeighbor.k_neighbors(mtx_np, k_user, numberOfUser)
    RatingPredict.rating_predict(numberOfUser, numberOfItem, neighbor_user, mtx_np, k_user, mtx_ds,ar)
