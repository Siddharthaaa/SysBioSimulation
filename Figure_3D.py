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

vpol_profile_vpols = np.logspace(0,3,30)

settings = np.zeros((2,2), dtype=object)
sim_series_vpol_ext = True

legend = False
legend_outside = True


if legend_outside:
    figsize=(6.5, 5.2)
    leg_loc = (1.15, 0.3)
else:
    figsize=(3.7, 5.2)
    leg_loc = "best"
    

#"Fig 3B2": # increasing
    
settings[1,0] = dict(
rbp_pos = 430,
rbp_inh = 1,
rbp_br_t= 1,
k1_t = 10,
k2_t = 1e-1,
k3_t =2e-2)
        
#"Fig 3B3": # bell shape
settings[0,0] = dict(
rbp_pos = 430,
rbp_br_t= 0.3,
rbp_inh = 1,
k1_t = 10,
k2_t = 0.2,
k3_t = 1)
    
#"Fig 3B4": # u-shape
settings[1,1] = dict(
rbp_pos = 455,
rbp_br_t= 2,
k1_t = 10,
k2_t = 0.2,
k3_t = 0.02,
rbp_inh = 1)
    
#"Fig 3B5": # 2 extremes 
settings[0,1] = dict(
rbp_pos = 455,
rbp_br_t= 0.5,
rbp_inh = 1,
k1_t = 10,
k2_t = 0.1,
k3_t = 2)


fig, axs = plt.subplots(2,2, figsize=(5,6), sharex=True, sharey=True)

s = bs.coTrSplMechanistic()

for i in range(len(settings)):
    for j in range(len(settings[i])):
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
        for s_i, vpol in enumerate(vpols):
            print("Sim count: %d" % s_i)
            s.set_param("vpol", vpol)
    #        s.simulate_ODE = True
            s.simulate(stoch=False, ODE=True)
    #        psi = s.get_psi_mean(ignore_fraction = 0.4)
            incl = s.get_res_col("Incl", method="ODE")[-1]
            skip = s.get_res_col("Skip", method="ODE")[-1]
            rbp_r = s.get_res_col("SkipInh", method="ODE")[-1]
            rbp_r += s.get_res_col("InclInh", method="ODE")[-1]
            rbp_reacts.append(rbp_r/(incl + skip))
            psi = incl/(incl+skip)
            psis.append(psi)
            ret = s.get_res_col("ret", method="ODE")[-1]
            rets.append(ret/s.init_state["P000"])
            
            
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
            
        
    #    fig, ax = plt.subplots()
        ax.plot(vpols, psis, lw=4, label = "PSI")
        if (sim_series_vpol_ext):
            ax.plot(vpols, psis_no_rbp, lw=2, ls=":", label = "PSI without RBP")
            ax.plot(vpols, psis_full_rbp, lw=2, ls=":", label = "PSI with 100% RBP")
    #        ax.plot(vpols, rets, lw=1, label="ret %")
            ax.plot(vpols, rbp_reacts, lw=1.5, ls="--", label="share of mRNA + RBP")
        
#        ax.set_xlabel("vpol [nt/s]")
#        ax.set_ylabel("PSI")
#        ax.set_title("rbp pos: %d, rbp_br: %.2f" % (s.params["rbp_pos"], s.params["rbp_br_t"]))
        ax.set_xscale("log")
if legend:
    axs[1,0].legend(bbox_to_anchor=(0., -0.50, 2., -0.3), loc='lower left',
           ncol=2)
axs[1,0].set_xlabel("vpol [nt/s]")
axs[1,1].set_xlabel("vpol [nt/s]")
axs[0,0].set_ylabel("PSI")
axs[1,0].set_ylabel("PSI")
#fig.tight_layout()
fig.legend(handles =ax.get_children()[0:3], ncol=2)
