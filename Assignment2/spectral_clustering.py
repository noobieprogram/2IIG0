import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

_datasets = [(noisy_circles,2), (noisy_moons,2), (varied, 3), (aniso,3)]


import numpy as np
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from sklearn.cluster import SpectralClustering
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import networkx as nx


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def sim_eps(D, k):
    """
    Returns the weighted adjacency matrix
    of the e-neighborhood graph of data matrix D
    """
    k = k[0]

    W = radius_neighbors_graph(D, k, mode='connectivity',
                           include_self=False).toarray()

    print("sym eps:", check_symmetric(W))

    return W

def sim_knn(D, k):
    """
    Returns the weighted adjacency matrix
    of the k-NN graph of data matrix D
    """

    k=k[1]
    
    W = kneighbors_graph(D, k, mode='connectivity',
                         include_self=False).toarray()
    W = 0.5 * (W + W.T)
    
    print("sym knn:",check_symmetric(W))
    
    return W

def laplacian(W):
    """
    Takes a weighted adjacency matrix and
    returns its symmetrix laplacian matrix
    """
    
    n = len(W)
    D = np.zeros((n,n), dtype=float) # initial 0 degree matrix

    # Compute the degree matrix
    for i in range(n):
        D[i, i] = np.sum(W[i])

    # L_sym = I-D^(-1/2)WD^(-1/2)
    L_sym = D[np.diag_indices(n)] = 1/ (D.diagonal()**0.5)
    L_sym = np.dot(D, W).dot(D)
    L_sym = np.identity(n)-L_sym
    
    print("sym laplacian:",check_symmetric(L_sym))

    return L_sym

# Good parameter values
# dataset_num: (eps, knn)
values = {
    0: (0.3, 80),
    1: (0.4, 100),
    2: (3.6, 250),
    3: (1.5, 40) # eps not good atm
}

def spectral_clustering(r, D, sim, laplacian, k):
    """
    Returns whatever a spectral clustering returns
    r: number of clusters
    D: data matrix
    sim: similarity function
    laplacian: symmetric laplacian function
    k: parameter value
    """
    n = len(D)
    
    W = sim(D, k) # weighted adjacency matrix of D, computed by the sim function
    L = laplacian(W) # symmetric laplacian of W
    w, v = np.linalg.eigh(L) # compute eig. values w and eig. vectors v
    y = kmeans2(v[:, 1:r], r)[1]
    
    plt.scatter(D[:, 0], D[:, 1], c=y, cmap='viridis')
    return y

i = 0
spectral_clustering(_datasets[i][1], _datasets[i][0][0], sim_knn, laplacian, k=values[i])