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
s.set_runtime(50000)
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
s.set_runtime(80000)
psis = []
res = []
ret = []
reti1 = []
reti2 = []
for v2  in np.linspace(0.001, 0.12, 201):
    s.set_param("u1_2_br", v2)
    s.set_param("u2_1_br", v2)
    s.simulate()
    psis.append(s.get_psi_mean())
    reti1.append(s.get_res_col("ret_i1"))
    reti2.append(s.get_res_col("ret_i2"))
    ret.append(s.get_res_col("ret"))
    
    res.append(s.get_res_from_expr("ret + ret_i1 + ret_i2"))

res = [np.mean(r[-2000:]) for r in res]
ret = [np.mean(r[-2000:]) for r in ret]
reti1 = [np.mean(r[-2000:]) for r in reti1]
reti2 = [np.mean(r[-2000:]) for r in reti2]

fig = plt.figure()
ax = fig.subplots()
ax.scatter(psis, res, label ="sum")
ax.scatter(psis, ret, label ="ret")
ax.scatter(psis, reti1, label ="ret_i1")
ax.scatter(psis, reti2, label ="ret_i2")

ax.set_xlabel("PSI")
ax.set_ylabel("#")
ax.set_title("By variation of v2")