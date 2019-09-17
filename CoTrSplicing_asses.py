#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:38:02 2019

@author: timur
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


def get_total_IR(s):
    r1 = s.get_res_col("ret")
    r2 = s.get_res_col("ret_i1")
    r3 = s.get_res_col("ret_i2")
    
    ir = r1 + r2 + r3
    
    return np.mean(ir[int(len(ir)/10):])
    
s = bs.get_exmpl_sim("CoTrSplicing")
#s.draw_pn(rates=False)
s.set_runtime(500)
s.show_interface()
s.compile_system()
v2s = np.linspace(0.1, 5, 40)
psis = []
ret_total = []
for v2 in v2s:
    s.set_param("u2_1_br", v2)
    s.set_param("u1_2_br", v2)
    s.simulate()
    psis.append(s.get_psi_mean())
    ret_total.append(get_total_IR(s))

s.plot_course(products=["ret_i2","ret_i1","ret"], res = ["stoch"])
s.plot_course(products=["Skip","Incl","ret", "mRNA"], res = ["stoch"])
s.plot_course(products=["Skip","Incl"], res = ["stoch"])
s.plot_course(products=["U1_1", "U1_2"], res = ["stoch"])
s.plot_course(products=["Pol_on", "U1_1"], res = ["stoch"])

fig = plt.figure()
ax = fig.subplots()
ax.plot(v2s, psis)

ax.set_xlabel("U2_1, U1_2 binding rate")
ax.set_ylabel("mean(PSI)")

fig = plt.figure()
ax = fig.subplots()

ax.plot(v2s, ret_total)
ax.set_xlabel("U2_1, U1_2 binding rate")
ax.set_ylabel("IR total")

    
s = bs.get_exmpl_sim("CoTrSplicing")
s.compile_system()
s.set_runtime(100000)
s.plot_par_var_1d("elong_v", np.linspace(20,200,10),None, s.get_psi_mean)

    
s = bs.get_exmpl_sim("CoTrSplicing")
s.set_runtime(40000)
pars = {"elong_v": np.linspace(10,500,50), "u1_2_br": np.linspace(0.01,0.4,40)}
res = s.plot_par_var_2d(pars, None, s.get_psi_mean, ignore_fraction=0.8)


s = bs.get_exmpl_sim("CoTrSplicing")
s.set_runtime(60000)
s.set_param("elong_v", 50)
psi_means =[]
psis = []
res = []
ret = []
reti1 = []
reti2 = []
incl =[]
skip = []
v2s = np.linspace(0.001, 0.12, 201)

for v2  in v2s:
    s.set_param("u1_2_br", v2)
    s.set_param("u2_1_br", v2)
    s.simulate()
    psi_means.append(s.get_psi_mean())
    psis.append(s.compute_psi()[1])
    
    reti1.append(s.get_res_col("ret_i1"))
    reti2.append(s.get_res_col("ret_i2"))
    ret.append(s.get_res_col("ret"))
    incl.append(s.get_res_col("Incl"))
    skip.append(s.get_res_col("Skip"))
    
    res.append(s.get_res_from_expr("ret + ret_i1 + ret_i2"))

res_means = [np.mean(r[-3000:]) for r in res]
res_stds = [np.std(r[-3000:]) for r in res]
ret_means = [np.mean(r[-3000:]) for r in ret]
reti1_means = [np.mean(r[-3000:]) for r in reti1]
reti2_means = [np.mean(r[-3000:]) for r in reti2]
psi_stds = [np.std(r) for r in psis]

incl_means = [np.mean(r[-3000:]) for r in incl]
incl_stds = [np.std(r[-3000:]) for r in incl]
skip_means = [np.mean(r[-3000:]) for r in skip]
skip_stds = [np.std(r[-3000:]) for r in skip]


fig = plt.figure()
ax = fig.subplots()
ax.scatter(psi_means, res_means, label ="sum")
ax.scatter(psi_means, ret_means, label ="ret")
ax.scatter(psi_means, reti1_means, label ="ret_i1")
ax.scatter(psi_means, reti2_means, label ="ret_i2")

ax.set_xlabel("PSI")
ax.set_ylabel("#")
ax.set_title("By variation of v2")
ax.legend()


fig = plt.figure()
ax = fig.subplots()

ax.plot(v2s, psi_means)
ax.set_xlabel("v2")
ax.set_ylabel("PSI")


fig = plt.figure()
ax = fig.subplots()
ax.scatter(psi_means, psi_stds)
ax.set_xlabel("PSI")
ax.set_ylabel("std(PSI)")

fig = plt.figure()
ax = fig.subplots()
ax.scatter(res_means, res_stds)
ax.set_xlabel("mean(total IR)")
ax.set_ylabel("std(total IR)")


fig = plt.figure()
ax = fig.subplots()

ax.scatter(incl_means, incl_stds, label="Incl")
ax.scatter(skip_means, skip_stds, label="Skip")
ax.set_xlabel("mean")
ax.set_ylabel("std")
ax.legend()


fig = plt.figure()
ax = fig.subplots()
ax.scatter(psi_means, incl_means, label = "Incl")
ax.scatter(psi_means, skip_means, label = "Skip")
ax.scatter(psi_means, [i+s for i,s  in zip(incl_means, skip_means)], label = "Sum")
ax.set_xlabel("mean(Psi)")
ax.set_ylabel("#")
ax.legend()


s = bs.get_exmpl_sim("CoTrSplicing")
s.set_runtime(80000)
s.set_param("u1_2_br", 0.05)
s.set_param("u2_1_br", 0.05)
v0s = np.linspace(10,130,41)
psis = []
for v0 in v0s:
    s.set_param("elong_v", v0)
    s.simulate()
    psis.append(s.get_psi_mean(ignore_fraction = 0.5))
    

fig = plt.figure()
ax = fig.subplots()
ax.plot(v0s, psis)
ax.set_xlabel("v0")
ax.set_ylabel("PSI")
ax.set_title("Elongation rate effect")
