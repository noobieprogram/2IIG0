import numpy as np
import random
from math import inf
from scipy.spatial.distance import cityblock, euclidean


def k_means(r: int, D, init: str, dist: str):
    X = initClusters(r, D, init, dist)

    # stopping criteria is convergence
    old_centroids = None
    while True:
        Y = clusterAssignments(X, D, dist)
        X = centroidsUpdate(Y, D)

        if X == old_centroids:
            break
        else:
            old_centroids = X

    return X, Y


def initClusters(r: int, D, init: str, dist="euclidean") -> np.ndarray:
    # we decided to make centroids of shape r * d instead of d * r for convenience

    if init == "random":
        centroids = np.zeros(shape=(r, len(D[0])))
        # find the maximum value for each dimension
        # in order to draw random samples from this space
        maxes = [0] * len(D[0])
        for data in D:
            for i in range(len(data)):
                if data[i] > maxes[i]:
                    maxes[i] = data[i]

        # generate r random points
        for i in range(r):
            sample = []
            for m in maxes:
                sample.append(random.uniform(0, int(m)))

            centroids[i] = np.array(sample)

        return np.array(centroids)

    elif init == "forgy":
        # choose r random points from D
        return random.sample(D, r)

    else:
        return k_meanspp(r, D, dist)


def k_meanspp(r: int, D, dist):
    s = 1
    X = random.choice(D)

    return None

# update the centroids
def centroidsUpdate(Y, D) -> np.ndarray:
    centroids = np.zeros(shape=(len(Y), len(D)))
    Y = np.matrix.transpose(Y)  # n*r matrix -> r*n matrix for convenience
    index = 0
    for cluster in Y:  # looping over each cluster
        this_cluster = []
        for j in range(len(cluster)):
            if cluster[j] == 1:
                this_cluster.append(j)  # D[j] was assigned to this cluster

        sum_points = [0] * len(D[0])
        for data in this_cluster:
            for i in range(len(D[data])):
                sum_points[i] += D[data][i]

        sum_points = [x / len(this_cluster) for x in sum_points]
        centroids[index] = np.array(sum_points)
        index += 1

    return centroids


def clusterAssignments(X, D, dist) -> np.ndarray:
    if dist == "euclidean":
        return euclidean_centroids(X, D)
    else:
        return manhattan_centroids(X, D)


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


X = np.array([
    [1, 3, 0],
    [4, 5, 0]
])

D = np.array([[3, 4, 0], [1, 2, 0], [1, 2, 3]])

print(random.choices(D, k=2))
