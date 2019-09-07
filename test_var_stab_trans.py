#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:02:14 2019

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
import glob
import math

s = bs.get_exmpl_sim()
s.set_runtime(10000)
s.set_raster_count(10001)

s.set_param("v_syn", 50)

psis = []
psi_means = []
for s1 in np.linspace(0.1,20, 50):
    s2 = 20.1 -s1
    s.set_param("s1", s1)
    s.set_param("s2", s2)
    
    s.simulate()
#    s.plot(psi_hist=True)
    psi = s.compute_psi()[1]
    psis.append(psi)
    psi_means.append(np.nanmean(psi))

psis = np.array(psis)
psi_means = np.array(psi_means)
    
plt.figure()
plt.plot(psi_means, np.nanstd(psis, axis=1))
plt.plot(psi_means, np.nanstd(np.arcsin(psis**0.5), axis=1))
