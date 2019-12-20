from data_ex3 import make_datasets

import numpy as np
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx

datasets = make_datasets()

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
    from data_ex3 import make_datasets

import numpy as np
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx

datasets = make_datasets()

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
    
    W = kneighbors_graph(D, k, mode='distance',
                         include_self=False).toarray()
    W = 0.5 * (W + W.T) # make matrix symmetric
    
    print("sym knn:",check_symmetric(W))
    print(W)
    
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
    1: (0.4, 120),
    2: (3.6, 250),
    3: (1.31, 232)
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
    # y = kmeans2(v[:, 1:r], r)[1]
    kmeans = KMeans(n_clusters=r, random_state=0).fit(v[:, 1:r])
    y = kmeans.labels_
    plt.scatter(D[:, 0], D[:, 1], c=y, cmap='viridis', alpha=0.5)
    
    return y

i = 3
spectral_clustering(datasets[i][1], datasets[i][0][0], sim_knn, laplacian, k=values[i])
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
    3: (1.31, 40)
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
    # y = kmeans2(v[:, 1:r], r)[1]
    kmeans = KMeans(n_clusters=r, random_state=0).fit(v[:, 1:r])
    y = kmeans.labels_
    plt.scatter(D[:, 0], D[:, 1], c=y, cmap='viridis')
    
    return y

i = 0
spectral_clustering(datasets[i][1], datasets[i][0][0], sim_knn, laplacian, k=values[i])