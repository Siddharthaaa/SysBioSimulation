#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:05:52 2019

@author: timur
"""


import numba as nb
import pandas as pd
import numpy as np
import glob
import os
import pylab as plt 
from sklearn.linear_model import LinearRegression
from bioch_sim import *
import bioch_sim as bs
import aux_th
import scipy as sp 
import random as rd
import matplotlib.cm as cm
from sklearn.decomposition import PCA

from support_th import *
if __name__ == "__main__":
    anz = 500
    psi_means = np.random.beta(4,4,size= anz)*0.4 + 0.3
    counts = np.random.randint(1,10, anz)
    tmp_compare_binomial_gillespie(counts,psi_means)
    tmp_compare_binomial_gillespie(counts,psi_means, exact_counts = True)
    
    indx = np.where(counts == 4)
    
#    sim = sim_tmp
#    i = indx[0][0]
#    sim.params = pars_tmp[i]
#    sim.results = results_tmp[i]
#    sim.plot_course(plot_psi=True)
#    
#    incl = sim.get_res_col("Incl")
#    skip = sim.get_res_col("Skip")
#    ges = incl + skip
#    fig = plt.figure()
#    plt.hist(incl + skip, bins = 20)
    ##s_res = s.compute_psi()
    ##s.plot_course()
    #
    #plt.scatter(counts, counts_tmp)
    #plt.scatter(psi_means, psis_tmp)