import matplotlib.pyplot as plt
import numpy as np
import random
from math import inf
from time import time
from scipy.spatial.distance import cityblock, euclidean
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import normalized_mutual_info_score as nmi


def k_means(r: int, D: np.ndarray, init: str, dist: str):
    X = initClusters(r, D, init, dist)
    old_centroids = None
    iterations = 0
    while True:
        iterations += 1
        Y, Y1 = clusterAssignments(X, D, dist)

        # plt.clf()
        # colors = ['#7f8c9b', '#68b1a7', '#e55ced', '#4b2e74', '#bcbb9e', '#f6f99e']
        # for key in Y:
        #     x = np.array(Y[key])[:, 0]
        #     y1 = np.array(Y[key])[:, 1]
        #     plt.scatter(x, y1, color=colors[key])
        #
        # # plot centroids
        # plt.scatter(X[:, 0], X[:, 1], color='#000000')
        # # save plot
        # name = "{}-{}-{}-{}".format(iterations, 'final', init, dist)
        # plt.savefig(
        #     "/Users/abdullahsaeed/OneDrive - TU Eindhoven/TU-e/Year 3/Data mining and machine learning 2IIG0/Assignment 2/{}.png".format(
        #         name))

        X = centroidsUpdate(Y, D, r)

        # stopping criterion is convergence
        if np.array_equal(X, old_centroids):
            break
        else:
            old_centroids = X

    print("Number of iterations = ", iterations)  # count the number of iterations it takes until convergences
    return X, Y, Y1


def initClusters(r: int, D, init: str, dist="euclidean") -> np.ndarray:
    # we decided to make centroids of shape r * d instead of d * r for convenience

    if init == "random":
        centroids = np.zeros(shape=(r, len(D[0])))
        # find the maximum value for each dimension
        # in order to draw random samples from this space
        maxes = [0] * len(D[0])
        for i in range(len(maxes)):
            maxes[i] = max(D[:, i])

        # generate r random points
        for i in range(r):
            sample = []
            for m in maxes:
                sample.append(random.uniform(0, int(m)))

            centroids[i] = np.array(sample)

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

    for key in Y:  # looping over each cluster
        # if this cluster has no points assigned to it, we skip
        if (len(Y[key])) == 0:
            continue
        # else we continue
        sum_points = [0] * len(D[0])
        for point in Y[key]:
            for i in range(len(point)):
                sum_points[i] += point[i]

        sum_points = [x / len(Y[key]) for x in sum_points]
        centroids[key] = np.array(sum_points)

    return centroids


def clusterAssignments(X, D, dist):
    # we decided to use Y as a dictionary as it is far more efficient than a sparse matrix
    Y = {i: [] for i in range(len(X))}

    # just to calculate NMI later
    Y1 = np.zeros(len(D), dtype=int)

    for i in range(len(D)):
        distance = inf
        cluster = 0
        for j in range(0, len(X)):
            if dist == "euclidean":
                dist = euclidean(np.array(D[i]), np.array(X[j]))
            else:
                dist = cityblock(np.array(D[i]), np.array(X[j]))

            if dist < distance:
                distance = dist
                cluster = j

        Y[cluster].append(D[i])
        Y1[i] = cluster

    return Y, Y1


def main():
    # generate data
    # D, y = make_blobs(n_samples=15000, centers=5, cluster_std=[3.9, 1.7, 1.5, 5.9, 2.8], n_features=2, random_state=10,
    #                   center_box=(-35.0, 25.0))
    D, y = make_blobs(n_samples=3500, cluster_std=[1.0, 2.5, 0.5], random_state=170, center_box=(-15.0, 5.0))

    # plot the final clusters
    inits = ["random", "forgy", "k-means++"]
    dists = ["euclidean", "manhattan"]

    for init in inits:
        for dist in dists:
            print("/-----------------------------------/")
            print("Dataset 1: {} initialization, {} distance".format(init, dist))

            # run k-means algorithm
            start = time()
            X, Y, Y1 = k_means(3, D, init, dist)
            print("Time taken = ", time() - start)

            nmi_score = nmi(y, Y1)
            print("NMI score = ", nmi_score)

            plt.clf()
            colors = ['#7f8c9b', '#68b1a7', '#e55ced', '#4b2e74', '#bcbb9e', '#f6f99e']
            for key in Y:
                x = np.array(Y[key])[:, 0]
                y1 = np.array(Y[key])[:, 1]
                plt.scatter(x, y1, color=colors[key])

            # plot centroids
            plt.scatter(X[:, 0], X[:, 1], color='#000000')
            # save plot
            name = "{}-{}-{}".format('final', init, dist)
            plt.savefig(
                "/Users/abdullahsaeed/OneDrive - TU Eindhoven/TU-e/Year 3/Data mining and machine learning 2IIG0/Assignment 2/{}.png".format(
                    name))


main()
