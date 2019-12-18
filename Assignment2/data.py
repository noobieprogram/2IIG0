from sklearn.datasets.samples_generator import make_blobs
import numpy as np

X, y = make_blobs(n_samples=15000, centers=5, cluster_std=[3.9, 1.7, 1.5, 5.9, 2.8], n_features=2, random_state=10,
                  center_box=(-35.0, 25.0))

X = np.vstack((X[y == 0][:5000], X[y == 1][:4500],
               X[y == 2][:4000], X[y == 3][:2000], X[y == 4][:1000]))
y = np.hstack((y[y == 0][:5000], y[y == 1][:4500],
               y[y == 2][:4000], y[y == 3][:2000], y[y == 4][:1000]))

# ---------------------------------- # old code
from scipy.spatial.distance import euclidean, cityblock
from math import inf


# calculate closest centroid for all points using
# manhattan distance
def manhattan_centroids(X, D):
    distance = inf
    cluster = 0
    Y = np.zeros(shape=(len(D), len(X)))
    for i in range(0, len(D)):
        for j in range(0, len(X)):
            dist = cityblock(np.array(D[i]), np.array(X[j]))
            if dist < distance:
                distance = dist
                cluster = j
        Y[i][cluster] = 1

    return Y


# calculate closest centroid for all points using
# euclidean distance
def euclidean_centroids(X, D):
    distance = inf
    cluster = 0
    Y = np.zeros(shape=(len(D), len(X)))
    for i in range(0, len(D)):
        for j in range(0, len(X)):
            dist = euclidean(np.array(D[i]), np.array(X[j]))
            if dist < distance:
                distance = dist
                cluster = j
        Y[i][cluster] = 1

    return Y
