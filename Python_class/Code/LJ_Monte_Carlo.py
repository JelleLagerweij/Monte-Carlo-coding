# -*- coding: utf-8 -*-
"""
Created on Thu May  5 23:28:15 2022

@author: Jelle
"""
import numpy as np
import pandas as pd
import scipy.constants as co
import scipy.optimize as opt
import sys as sys


class State:
    def __init__(self, T, rho, m_a, eps, sig, N_or_file):
        self.T = T
        self.rho = rho
        self.m_a = m_a
        self.eps = eps*co.k
        self.sig = sig*1e-10
        self.tail = 0
        self.shift = 0
        self.initialConfiguration(N_or_file)

    def initialConfiguration(self, N_or_file):
        # initialise positions, either from data file or random distribution.
        if type(N_or_file) is str:
            data = pd.read_csv(N_or_file, delim_whitespace=True, header=None,
                               skiprows=2)
            self.x = (np.array((data[1], data[2], data[3])).T)*1e-10
            self.L = np.round(np.max(np.abs(self.x*1e10)))*1e-10
            self.Rcut = self.L/2
            self.rho = self.m_a*self.x.shape[0]/(1000*co.N_A*self.L**3)

        elif type(N_or_file) is int:
            self.L = np.power(self.m_a*N_or_file/(co.N_A*1000*self.rho), 1/3)
            self.Rcut = self.L/2
            self.x = np.random.rand(N_or_file, 3)*self.L

        else:
            raise ValueError("N_or_file has to be the ammount of particles",
                             "int, or the file location with an initial state")

    def modelCorrections(self, Rcut=14, Tail=False, Shift=False):
            self.Rcut = Rcut
            
            if Shift is True:
                self.shift = 4*self.eps*((self.sig/self.Rcut)**12 - (self.sig/self.Rcut)**6)
                
            if Tail is True:
                self.tail = 0  # TODO here as otherwise no tail correction implemented
        
    def allDistances(self):
        x = self.x
        L = self.L
        N, D = x.shape

        # Then compute the MIC using PBC.
        r = np.broadcast_to(x, (N, N, D))
        rel_r = (r - r.transpose(1, 0, 2) + L/2) % L - L/2
        d_sq = np.einsum('ijk, ijk->ij', rel_r, rel_r, optimize='optimal')
        np.fill_diagonal(d_sq, np.inf)  # Removes self imaging
        self.d_sq = d_sq  # the squared distances that are returned

    def totalEnergy(self):
        d_sq = np.copy(self.d_sq)
        N = d_sq.shape[0]
        Rcut = self.Rcut
        eps = self.eps
        sig = self.sig
        tail = self.tail
        shift = self.shift

        d_sq[d_sq > Rcut**2] = np.inf  # Applies the cut off distance.
        sr6 = sig**6/(d_sq*d_sq*d_sq)  # is faster than power
        sr12 = sr6*sr6
        n = np.count_nonzero(sr6)  # count all particles inside Rcut
        Etot = 2*eps*np.sum(sr12-sr6) - n*shift/2 + N*tail
        self.Etot = Etot
        return Etot

    def pressure(self):
        T = self.T
        sig = self.sig
        eps = self.eps
        L = self.L
        V = L**3
        d_sq = self.d_sq
        sr2 = sig**2/d_sq
        sr6 = sr2*sr2*sr2  # is faster than power
        sr12 = sr6*sr6
        P = co.k*T*d_sq.shape[0]/V - 24*eps*np.sum(sr6 - 2*sr12)/(6*V)
        self.P = P
        return P

    def singleEnergy(self, Ni, trial=False):
        sig = self.sig
        eps = self.eps
        tail = self.tail
        shift = self.shift

        if trial is True:
            x = self.x_trial
        else:
            x = self.x

        L = self.L
        Rcut = self.Rcut
        N, D = x.shape

        # Then compute the MIC using PBC
        rel_r = (x - x[Ni, :] + L/2) % L - L/2
        d_sq = np.einsum('ij, ij -> i', rel_r, rel_r)
        d_sq[Ni] = np.inf  # Removes self imaging
        d_sq[d_sq > Rcut**2] = np.inf  # Applies the cut off distance.

        sr_2 = sig**2/d_sq
        sr6 = sr_2*sr_2*sr_2   # atractive part of the LJ potential
        sr12 = sr6*sr6  # repulsive part of the LJ potential (Pauli)
 
        n = np.count_nonzero(sr6)  # number of particles inside Rcut
        E_single = 4*eps*np.sum(sr12 - sr6) - n*shift + tail
        return E_single
    
    def newStepsize(self):
        if self.acceptance < 0.2:
            self.max_step *= 0.7
        elif self.acceptance < 0.4:
            self.max_step *= 0.95
        elif self.acceptance > 0.8:
            self.max_step /= 0.7
        elif self.acceptance > 0.5:
            self.max_step /= -.95
        
        if self.max_step > self.L/2:
            self.max_step = self.L


def monteCarlo(state, n, max_step_init, startup_eq=True):
    """
    This is the main function to execute the canonecal ensemble Monte Carlo. It
    performs the trial moves and computes the observables.

    Parameters
    ----------
    state : object
        The object that contains all characteristics of the current state of
        the ensemble. It hase to be of the class State.
    n : integer
        The ammount of times that the observables are sampled. Between every
        sample 2N trial moves are performed..
    max_step_init : float
        The initial maximum step size. The value of this parameter should be in
        he correct ballpark, however it will be optimised along the way.
    startup_eq : boolean, optional
        If the model is run with an already equilibrated dataset, the
        equilibration phase of the simulation can be turned of (False). The
        default is True.

    Returns
    -------
    E_tot : tuple of 2 floats
        The average total potential energy and its estimated error.
    P : tuple of 2 floats
        The average pressure of the system and its estimated error.
    rad_dis_m : array of floats
        The average radial distribution function.
    r : array of floats
        The positions at which the datapoints of the average radial
        distribution function are sampled.

    """
    state.max_step = max_step_init*1e-10
    x = state.x
    L = state.L
    N, D = x.shape

    stepsize_startup = np.zeros(1000)
    acceptance_rate = np.zeros(1000)
    if startup_eq is True:
        for j in range(1000):
            state.accept = 0
            for i in range(int(N)):
                translate(state)
            state.acceptance = state.accept/(int(N))
            acceptance_rate[j] = state.acceptance
            stepsize_startup[j] = state.max_step
            state.newStepsize()

    state.stepsize_startup = stepsize_startup
    state.acceptance_rate = acceptance_rate

    # Now real measurement data is generated
    E_tot = np.zeros(n)
    P = np.zeros(n)
    rad_dis = np.zeros((1000, n))
    state.allDistances()
    r = np.histogram(np.sqrt(state.d_sq), bins=1000, range=(0, L/2))[1]
    r = r[:-1] + (r[1]-r[0])/2  # remove last bin edge and shift to center bins

    for i in range(n):
        state.accept = 0
        for j in range(N):
            translate(state)
        state.acceptance = state.accept/(N)
        state.allDistances()
        E_tot[i] = state.totalEnergy()
        P[i] = state.pressure()
        n_r = np.histogram(np.sqrt(state.d_sq), bins=1000, range=(0, L/2))[0]
        rad_dis[:, i] = (L**3*n_r)/(N*(N-1)*4*np.pi*(r**2)*(r[1]-r[0]))
        # state.progress = i/n
        # update_progress(state.progress)

    # state.progress = 1
    # update_progress(state.progress)

    state.trial_moves = 2*N*np.arange(n)
    state.E_tot = E_tot
    state.P = P
    state.rad_dis = rad_dis
    state.r_bins = r

    E_tot = statistics(E_tot)
    P = statistics(P)
    rad_dis_m = np.zeros((rad_dis.shape[0], 2))
    for i in range(rad_dis.shape[0]):
        rad_dis_m[i, :] = statistics(rad_dis[i, :])
    return E_tot, P, rad_dis_m, r


def translate(state):
    x = state.x
    N, D = x.shape
    T = state.T
    L = state.L
    max_step = state.max_step

    # Determine trial move
    Ni = np.random.randint(0, high=N)  # select the particle to trial step
    trial_move = 2*max_step*np.random.rand(1, 3) - 1*max_step
    
    #####
    x_trial = np.copy(x)  # copy to avoid edditing the origional one yet
    x_trial[Ni, :] = (x_trial[Ni, :] + trial_move) % L  # PBC applies
    state.x_trial = x_trial
    #####

    # Compute the energies and the difference between them
    U = state.singleEnergy(Ni)
    U_trial = state.singleEnergy(Ni, trial=True)
    dU = U_trial - U

    # To avoid overflow in expontentials when starting up, asses D_U/T<0 apart.
    if dU < 0:
        state.x = state.x_trial
        state.accept = state.accept + 1
    else:
        p_acc = np.exp(-dU/(co.k*T))
        if p_acc > np.random.rand():
            state.x = state.x_trial
            state.accept = state.accept + 1


def statistics(s):
    # Split the array into blocks
    num_blocks = 5
    s_blocks = np.array_split(s, num_blocks)
    
    # Calculate the mean of each block
    block_means = [np.mean(block) for block in s_blocks]
    
    # Calculate the mean and standard error of the block means
    mean = np.mean(block_means)
    error = np.std(block_means, ddof=1) / np.sqrt(num_blocks)
    
    return (mean, error)

def update_progress(progress):
    """
    this is just the progress report code of some source to make our result
    somewhat clear. The source is:
    https://stackoverflow.com/questions/3160699/python-progress-bar
    """
    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), np.round(progress*100, 3), status)
    sys.stdout.write(text)
    sys.stdout.flush()

