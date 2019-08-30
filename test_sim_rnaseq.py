#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:55:07 2019

@author: timur


"""


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import scipy.stats as st
import pyabc as pa
import pandas as pd
import support_th as sth
import bioch_sim as bs

class VectorDistance(pa.Distance):
    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        
        l_sum = 0
        for k in x.keys():
            l = np.linalg.norm(x[k] - x_0[k])
            l_sum += l
        return l_sum
    
distance = VectorDistance()

counts = np.array([5,10,15,20,25])
psis = np.linspace(0,1,5)

psi_stds = st.binom.std(counts, psis)/counts

pars = np.linspace(0.1, 1, 30)
distances = []
for p in pars:
    stds = sth.tmp_simulate_std_gillespie(counts, psis, runtime=10000,
                                          sim_rnaseq=p,
                                          extrapolate_counts=p)
    distances.append(distance({"x": psi_stds}, {"x": stds}))

plt.plot(pars, distances)