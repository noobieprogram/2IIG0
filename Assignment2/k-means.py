import matplotlib.pyplot as plt
import numpy as np
import random
from math import inf
from scipy.spatial.distance import cityblock, euclidean
from sklearn.datasets.samples_generator import make_blobs


def plot(where, init, dist, X, clear=False, D=None):
    Xx = X[:, 0]
    Xy = X[:, 1]
    if clear:
        plt.clf()
        plt.scatter(D[:, 0], D[:, 1])
    plt.scatter(Xx, Xy)
    name = "{}-{}-{}".format(where, init, dist)
    plt.savefig(
        "/Users/abdullahsaeed/OneDrive - TU Eindhoven/TU-e/Year 3/Data mining and machine learning 2IIG0/Assignment 2/{}.png".format(
            name))


def k_means(r: int, D: np.ndarray, init: str, dist: str):
    X = initClusters(r, D, init, dist)
    plot('initial', init, dist, X)
    old_centroids = None
    iterations = 0
    while True:
        print(iterations)  # count the number of iterations it takes until convergences
        iterations += 1
        Y = clusterAssignments(X, D, dist)
        X = centroidsUpdate(Y, D, r)

        # stopping criterion is convergence
        if np.array_equal(X, old_centroids):
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

        print(len(centroids) == r)

        return np.array(centroids)

    elif init == "forgy":
        # choose r random points from D
        return np.array(random.sample(D.tolist(), r))

    else:
        return k_meanspp(r, D, dist)


def k_meanspp(r: int, D: np.ndarray, dist: str) -> np.ndarray:
    C = [random.choice(D)]

    for k in range(1, r):
        if dist == "euclidean":
            D2 = np.array([min([euclidean(c, x) for c in C]) for x in D])
        else:
            D2 = np.array([min([cityblock(c, x) for c in C]) for x in D])
        probs = D2 / D2.sum()
        cumulative_prob = probs.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_prob):
            if r < p:
                i = j
                break
        C.append(D[i])

    return np.array(C)


# update the centroids
def centroidsUpdate(Y, D, r: int) -> np.ndarray:
    centroids = np.zeros(shape=(r, len(D[0])))
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

        # if this cluster has no points assigned to it, we continue
        if (len(this_cluster)) == 0:
            continue
        sum_points = [x / len(this_cluster) for x in sum_points]
        centroids[index] = np.array(sum_points)
        index += 1

    return centroids


def clusterAssignments(X, D, dist) -> np.ndarray:
    # probably more efficient to use dictionaries but assignment wants to use a matrix
    Y = np.zeros(shape=(len(D), len(X)), dtype=int)

    for i in range(len(D)):
        distance = inf
        cluster = 0
        for j in range(len(X)):
            if dist == "euclidean":
                dist = euclidean(np.array(D[i]), np.array(X[j]))

            else:
                dist = cityblock(np.array(D[i]), np.array(X[j]))

            if dist < distance:
                distance = dist
                cluster = j
        Y[i][cluster] = 1

    return Y


def main():
    # generate data
    D, y = make_blobs(n_samples=15000, centers=5, cluster_std=[3.9, 1.7, 1.5, 5.9, 2.8], n_features=2, random_state=10,
                      center_box=(-35.0, 25.0))
    # X = np.vstack((X[y == 0][:5000], X[y == 1][:4500],
    #                X[y == 2][:4000], X[y == 3][:2000], X[y == 4][:1000]))
    # y = np.hstack((y[y == 0][:5000], y[y == 1][:4500],
    #                y[y == 2][:4000], y[y == 3][:2000], y[y == 4][:1000]))
    D2, y2 = make_blobs(n_samples=3500, cluster_std=[1.0, 2.5, 0.5], random_state=170, center_box=(-15.0, 5.0))

    # plot generated data and save to disk
    init = "forgy"
    dist = "euclidean"
    plot('raw1', init, dist, D)

    # run k-means algorithm
    r = 5
    X, Y = k_means(r, D, init, dist)

    # plot the final centroids and save
    plot('final', init, dist, X, True, D)


main()
