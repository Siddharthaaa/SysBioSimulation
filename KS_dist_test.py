# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:16:42 2019

@author: timuhorn
"""

import numpy as np
import scipy as sp
import scipy.stats as sts
import matplotlib.pyplot as plt

d1 = sp.random.normal(0, 0.5, 10000) 
d2 = sp.random.normal(2,1,10000)

d_t1  = np.hstack((d1,d2))

d1 = sp.random.normal(0, 0.5, 1000) 
d2 = sp.random.normal(2,1,1000)

d_t2  = np.hstack((d1,d2))

sts.ks_2samp(d_t1,d_t2)

plt.hist(d_t2)
plt.hist(d_t1)

sts.ks_2samp(d_t2, d_t1)
