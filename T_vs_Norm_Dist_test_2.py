
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:04:01 2019

Discription: simulation of t-distibution 
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
sample_size = 10
sample_size2 = 15
sample_count = 200000

sigma = 3
mu = 4

samples = np.random.normal(mu,sigma,(sample_count,sample_size))
samples2 = np.random.normal(mu,sigma,(sample_count,sample_size2))


std_tmp = np.std(samples, axis=1, ddof = 1)
mean1  = np.mean(samples, axis=1)
std1 = [np.sqrt((((samp-mean)**2)).sum()/(len(samp)-1)) for samp, mean in zip(samples, mean1)]

res1 = (mean1 - mu)/std1 *np.sqrt(sample_size) 

#res1 = np.mean(samples, axis=1)/np.std(samples, axis=1)
res2 = np.mean(samples2, axis=1)/np.std(samples2, axis=1)



x = np.linspace(-np.std(res1)*4,np.std(res1)*4,100)
#(n, bins, patches) = plt.hist((res1, res2), 50, label = ("size: %d" % sample_size, "size: %d" % sample_size2))
(n, bins, patches) = plt.hist(res1, 500, label = "size: %d" % sample_size)
#plt.hist(np.random.standard_t(7, 200000), 500)

plt.plot(x, sts.norm.pdf(x, np.mean(res1), 1) *(bins[1]-bins[0]) * len(res1.flatten()), "r--", label= "norm.pdf")
plt.plot(x, sts.t.pdf(x, sample_size-1, np.mean(res1), 1) *(bins[1]-bins[0]) * len(res1.flatten()), "g--", label= "t.pdf")
plt.legend()


plt.title("Sample size: %d" % sample_size)

