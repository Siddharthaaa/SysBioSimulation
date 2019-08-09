# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:04:01 2019

@author: timuhorn
"""

import numpy as np
import scipy as sp
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.stats as stats

def z_trans(arr, constr = 0):
    res = np.array(arr)
    i=0
    for row in arr:
        mu = np.mean(row)
        row_shift = row-mu
        sigma = np.sqrt((row_shift**2).sum() / (len(row)-constr))
#        print(sigma)
        res[i,:] = row_shift / sigma
        i+=1;
    return res
sample_size = 3
sample_count = 10000

samples = np.random.normal(5,3,(sample_count,sample_size))

x = np.linspace(-3,3,100)

res1 = z_trans(samples, 0)
res2 = z_trans(samples, 1)

res3 = z_trans(res2,0)

(n, bins, patches) = plt.hist((res1.flatten(), res2.flatten()), 20, label = ("FG:n", "FG:n-1"))


plt.plot(x, sts.norm.pdf(x, 0, 1) *(bins[1]-bins[0]) * len(res1.flatten()), "r--")
plt.legend()

plt.title("Sample size: %d" % sample_size)