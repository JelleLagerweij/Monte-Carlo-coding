# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:17:08 2022

@author: Jelle
"""
import numpy as np
import scipy.spatial as spatial


def allDistances1(x, L):
    N, D = x.shape

    # Then compute the MIC using PBC.
    r = np.broadcast_to(x, (N, N, D))
    rel_r = (r - r.transpose(1, 0, 2) + L/2) % L - L/2
    d_sq = np.einsum('ijk, ijk->ij', rel_r, rel_r, optimize='optimal')
    np.fill_diagonal(d_sq, np.inf)  # Removes self imaging
    return d_sq  # the squared distances that are returned


def allDistances2(x, L):
    N, D = x.shape
    r = (spatial.distance_matrix(x, x) + L/2) % L - L/2
    np.fill_diagonal(r, np.inf)  # Removes self imaging
    return r

# using ckdtree query on the j9 long array
def allDistances4(x, L):
    N, D = x.shape
    tree=spatial.cKDTree(L)
    dist,minid=tree.query(x)
    return(dist.reshape([N,N]))

L = 30*1e-10
x = np.random.rand(500, 3)*L