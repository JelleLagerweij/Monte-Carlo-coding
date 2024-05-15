# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:17:08 2022

@author: Jelle
"""
import numpy as np
import scipy.spatial as spatial

# Test Calculating distances from every particle i to j

def allDistances1(x, L):
    N, D = x.shape
    d = np.zeros((N, N))

    for i in range(N):
        pos_ix = x[i, 0]
        pos_iy = x[i, 1]
        pos_iz = x[i, 2]
        
        for j in range(N):
            pos_jx = x[j, 0]
            pos_jy = x[j, 1]
            pos_jz = x[j, 2]
            
            dx = pos_jx - pos_ix
            dy = pos_jy - pos_iy
            dz = pos_jz - pos_iz
            
            d[i, j] = np.sqrt(dx**2 + dy**2 + dz**2)
    return d  # the squared distances that are returned


def allDistances2(x, L):
    N, D = x.shape
    d_sq = np.zeros((N, N))

    for i in range(N):
        pos_i = x[i, :]
        
        for j in range(i+1, N):
            pos_j = x[j, :]
            
            d = pos_j - pos_i
            
            d_sq[i, j] = d[0]**2 + d[1]**2 + d[2]**2
    return d_sq  # the squared distances that are returnedhe squared distances that are returned


def allDistances3(x, L,):
    N, D = x.shape
    d_sq = np.zeros((N, N))
    
    for (i, pos_i) in enumerate(x):  # loop over all rows of 2D array X
        d = x - pos_i  # subtrack pos_i immidiately of all items.
        d_sq[i, :] = d[0]**2 + d[1]**2 + d[2]**2

    return d_sq  # the squared distances that are returned


def allDistances5(x, L):
    N, D = x.shape

    # Then compute the MIC using PBC.
    r = np.broadcast_to(x, (N, N, D))
    rel_r = r - r.transpose(1, 0, 2)
    d_sq = np.einsum('ijk, ijk->ij',
                     rel_r, rel_r,
                     optimize='optimal')
    return d_sq  # the squared distances that are returned


def allDistances6(x, L):
    # Use the scipy.spation package
    N, D = x.shape
    r = spatial.distance_matrix(x, x)
    np.fill_diagonal(r, np.inf)  # Removes self imaging
    return r


# using ckdtree query on the j9 long array
def allDistances7(x, L):
    N, D = x.shape
    tree=spatial.cKDTree(L)
    dist,minid=tree.query(x)
    return(dist.reshape([N,N]))

L = 30
x = np.random.rand(500, 3)*L