#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:59:30 2020

@author: timur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:20:54 2019

@author: timur
"""



import bioch_sim as bs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np
import sympy as sy
from sympy.parsing.sympy_parser import parse_expr

vpol_profile_vpols = np.logspace(0,3,50)

settings = np.zeros((2,2), dtype=object)
sim_series_vpol_ext = True
    

if figure == "Fig 3B2": # increasing
    
    settings[1,0] = dict(
    rbp_pos = 300,
    rbp_inh = 1,
    rbp_br_t= 1,
    k1_t = 10,
    k2_t = 1e-1,
    k3_t =2e-2)
        
if figure == "Fig 3B3": # bell shape
    settings[0,0] = dict(
    rbp_pos = 300,
    rbp_br_t= 0.4,
    rbp_inh = 1,
    k1_t = 10,
    k2_t = 3e-1,
    k3_t = 3)
    
if figure == "Fig 3B4": # u-shape
    settings[1,1] = dict(
    rbp_pos = 300,
    rbp_br_t= 2,
    k1_t = 10,
    k2_t = 5e-1,
    k3_t = 2e-1,
    rbp_inh = 0.98)
    
if figure == "Fig 3B5": # 2 extremes 
    settings[1,0] = dict(
    rbp_pos = 300,
    rbp_br_t= 1,
    rbp_inh = 0.98,
    k1 = 10,
    k2 = 4e-1,
    k3 = 2)
#    
#if figure == "Fig 3B6": # 2 extremes
#    sim_series_vpol = True
#    vpol_profile_rbp_pos = 220
#    vpol_profile_vpols = np.logspace(0,3,50)
#    rbp_br= 1
#    rbp_inh = 0.98
#    k1 = 0.5
#    k2 = 0.2
#    k3 = 0.4
#    sim_series_vpol_ext = True
#    

fig, ax = plt.subplots(2,2, figsize=(5,5))

s = bs.coTrSplMechanistic()

for i in len(settings):
    for j in len(settings[i]):
        ax = axs[i,j]
    vpols = vpol_profile_vpols
    s.set_raster(30001)
    s.set_runtime(1e5)
    s.params.update(settings[i,j])
    psis = []
    psis_no_rbp = []
    psis_full_rbp = []
    rets = []
    rbp_reacts = []
    _rbp_br = s.params["rbp_br_t"]
    for i, vpol in enumerate(vpols):
        print("Sim count: %d" % i)
        s.set_param("vpol", vpol)
#        s.simulate_ODE = True
        s.simulate(stoch=False, ODE=True)
#        psi = s.get_psi_mean(ignore_fraction = 0.4)
        incl = s.get_res_col("Incl", method="ODE")[-1]
        skip = s.get_res_col("Skip", method="ODE")[-1]
        rbp_r = s.get_res_col("TEST_SKIP", method="ODE")[-1]
        rbp_r += s.get_res_col("TEST_INCL", method="ODE")[-1]
        rbp_reacts.append(rbp_r/(incl + skip))
        psi = incl/(incl+skip)
        psis.append(psi)
        ret = s.get_res_col("ret", method="ODE")[-1]
        rets.append(ret/init_mol_count)
        
        s.set_param("rbp_br_t", 0)
        s.simulate(stoch=False, ODE=True)
        incl = s.get_res_col("Incl", method="ODE")[-1]
        skip = s.get_res_col("Skip", method="ODE")[-1]
        psi = incl/(incl+skip)
        psis_no_rbp.append(psi)
        
        s.set_param("rbp_br_t", 1e6)
        s.simulate(stoch=False, ODE=True)
        incl = s.get_res_col("Incl", method="ODE")[-1]
        skip = s.get_res_col("Skip", method="ODE")[-1]
        psi = incl/(incl+skip)
        psis_full_rbp.append(psi)
        
        s.set_param("rbp_br_t", _rbp_br)
        
    
    fig, ax = plt.subplots()
    ax.plot(vpols, psis, lw=4, label = "PSI")
    if (sim_series_vpol_ext):
        ax.plot(vpols, psis_no_rbp, lw=2, ls=":", label = "PSI without RBP")
        ax.plot(vpols, psis_full_rbp, lw=2, ls=":", label = "PSI with 100% RBP")
#        ax.plot(vpols, rets, lw=1, label="ret %")
        ax.plot(vpols, rbp_reacts, lw=1.5, ls="--", label="share of mRNA + RBP")
    
    ax.set_xlabel("vpol [nt/s]")
    ax.set_ylabel("PSI")
    ax.set_title("rbp pos: %d, rbp_br: %.2f" % (rbp_pos, s.params["rbp_br_t"]))
    ax.set_xscale("log")
    if sim_series_vpol_legend:
        ax.legend()
    else:
        ax.legend(loc = (1.1, 0.2))
    
