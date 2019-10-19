import numpy as np
import pandas as pd
import scipy.sparse as sparse
from datetime import datetime
from pandas import DataFrame
from pandas import Series
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import  warnings



def get_neighbors(mdx, k, numberOfUser):

    warnings.filterwarnings("ignore")

    # Using PCA for dimensionality reduction
    start = datetime.now()
    pca = PCA(n_components=200, copy=True)
    pca_mtx = pca.fit_transform(mdx)
    end = datetime.now()
    print(start, end)



    # Build the KNN model
    start = datetime.now()
    neigh = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric="cosine", n_jobs=1)
    neigh.fit(pca_mtx)
    end = datetime.now()
    print(start, end)

    # Obtain the k nearst neighbors for each user

    start = datetime.now()
    distance, neighbor = neigh.kneighbors(pca_mtx)
    print(start, end)




    ulist = list()
    dlist = list()
    nlist = list()

    for user in range(1, numberOfUser):
        for i in range(1, k + 1):
            ulist.append(user)
        dlist += list(distance[user][1:])
        nlist += list(neighbor[user][1:])

    df_neighbor = DataFrame(columns=['user', 'neighbor', 'distance'])
    df_neighbor['user'] = ulist
    df_neighbor['neighbor'] = nlist
    df_neighbor['distance'] = dlist


    df_neighbor['similarity'] = 0.5 + 0.5 * df_neighbor['distance']
    df_neighbor.to_csv('user_neighbors.csv', index=False)
    new_neighbor = []
    for i in range(len(neighbor)):
        new_neighbor.append(neighbor[i][1:])
    return new_neighbor


def k_neighbors(mdx, k_user,numberOfUser):
    neighbor_user = get_neighbors(mdx, k_user, numberOfUser)


    return neighbor_user
