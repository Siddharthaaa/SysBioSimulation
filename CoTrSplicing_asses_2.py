# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 09:53:16 2019

@author: Timur
"""



from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import scipy.stats as st
#import pyabc as pa
import pandas as pd
import support_th as sth
import bioch_sim as bs
import glob
    


last_n = 5000

s = bs.get_exmpl_sim("CoTrSplicing")
s.set_runtime(2e5)
s.set_param("d1",0.0003)
s.set_param("d2",0.0003)
s.set_param("u2ur", 0.02)
s.set_param("u1ur", 0.02)

psi_means =[]
psis = []
res = []
ret = []
reti1 = []
reti2 = []
incl =[]
skip = []
v2s = np.logspace(-3, 0, 201)
v0s = np.linspace(10,200,201)
#for v2  in v2s:
#    s.set_param("u1_2_br", v2)
#    s.set_param("u2_1_br", v2)
#    s.simulate()
#    psi_means.append(s.get_psi_mean())
#    psis.append(s.compute_psi(ignore_fraction=0.5)[1])
#    
#    reti1.append(s.get_res_col("ret_i1"))
#    reti2.append(s.get_res_col("ret_i2"))
#    ret.append(s.get_res_col("ret"))
#    incl.append(s.get_res_col("Incl"))
#    skip.append(s.get_res_col("Skip"))
#    
#    res.append(s.get_res_by_expr("ret + ret_i1 + ret_i2"))

for v0  in v0s:
    s.set_param("elong_v", v0)
#    s.set_param("u2_1_br", v2)
    s.simulate()
    psi_means.append(s.get_psi_mean())
    psis.append(s.compute_psi(ignore_fraction=0.5)[1])
    
    reti1.append(s.get_res_col("ret_i1"))
    reti2.append(s.get_res_col("ret_i2"))
    ret.append(s.get_res_col("ret"))
    incl.append(s.get_res_col("Incl"))
    skip.append(s.get_res_col("Skip"))
    
    res.append(s.get_res_by_expr("ret + ret_i1 + ret_i2"))

res_means = [np.mean(r[-last_n:]) for r in res]
res_stds = [np.std(r[-last_n:]) for r in res]
ret_means = [np.mean(r[-last_n:]) for r in ret]
reti1_means = [np.mean(r[-last_n:]) for r in reti1]
reti2_means = [np.mean(r[-last_n:]) for r in reti2]
psi_stds = [np.std(r[-last_n:]) for r in psis]
psi_stds_stabl = [np.std(np.arcsin(np.sqrt(r[-last_n:]))) for r in psis]

incl_means = [np.mean(r[-last_n:]) for r in incl]
incl_stds = [np.std(r[-last_n:]) for r in incl]
skip_means = [np.mean(r[-last_n:]) for r in skip]
skip_stds = [np.std(r[-last_n:]) for r in skip]

counts = np.add(incl_means, skip_means)

psi_std_basic_model = sth.tmp_simulate_std_gillespie(counts, psi_means, var_stab=False)
psi_std_basic_model_stabl = sth.tmp_simulate_std_gillespie(counts, psi_means, var_stab=True)


fig = plt.figure()
ax = fig.add_subplot(1,3,1)
ax.scatter(psi_stds, psi_std_basic_model)
ax.set_xlabel("CoTranscr. Splicing: std(PSI)")
ax.set_ylabel("Basic Spl. model: std(PSI)")
ax.set_title("Without stabilizing")
std_max = max(psi_std_basic_model)
std_max *= 1.1
ax.plot([0,std_max], [0, std_max], c = "red")

ax = fig.add_subplot(1,3,2)
ax.scatter(psi_stds_stabl, psi_std_basic_model_stabl)
ax.set_xlabel("CoTranscr. Splicing: std(PSI)")
ax.set_ylabel("Basic Spl. model: std(PSI)")
ax.set_title("With stabilizing")
std_max = max(psi_std_basic_model)
std_max *= 1.1
ax.plot([0,std_max], [0, std_max], c = "red")


ax = fig.add_subplot(1,3,3)
ax.set_title("counts dependency")
ax.set_xlabel("counts")
ax.set_ylabel("$\Delta STD(PSI)$")
ax.scatter(counts, np.subtract(psi_stds,psi_std_basic_model))

fig.suptitle("Variation of v0")