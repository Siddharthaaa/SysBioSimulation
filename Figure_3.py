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


import os
import bioch_sim as bs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np
import sympy as sy

vpol_profile_vpols = np.logspace(0,3,40)
sim_series_vpol_ext = True
legend = False
legend_outside = True

parameters_plot = True
inhibition_plot = True


parameters_table_dir = os.path.join("docs", "pars_csv")
sbml_dir = os.path.join("docs", "sbml")
create_model_files = True

if legend_outside:
    figsize=(6.5, 5.2)
    leg_loc = (1.15, 0.3)
else:
    figsize=(3.7, 5.2)
    leg_loc = "best"

settings = np.zeros((2,2), dtype=object)    


f_names = np.zeros((2,2), dtype=object)


#"Fig 3B2": # increasing    
settings[1,0] = dict(
rbp_pos = 430,
rbp_inh = 1,
rbp_br_t= 1,
k1_t = 10,
k2_t = 1e-1,
k3_t =2e-2)
f_names[1,0] = "Fig.3D_increasing"
        
#"Fig 3B3": # bell shape
settings[0,0] = dict(
rbp_pos = 430,
rbp_br_t= 0.3,
rbp_inh = 1,
k1_t = 10,
k2_t = 0.2,
k3_t = 1)
f_names[0,0] = "Fig.3D_bell_shape"

    
#"Fig 3B4": # u-shape
settings[1,1] = dict(
rbp_pos = 455,
rbp_br_t= 2,
k1_t = 10,
k2_t = 0.2,
k3_t = 0.02,
rbp_inh = 1)
f_names[1,1] = "Fig.3D_u_shape"

    
#"Fig 3B5": # 2 extremes 
settings[0,1] = dict(
rbp_pos = 455,
rbp_br_t= 0.5,
rbp_inh = 1,
k1_t = 10,
k2_t = 0.1,
k3_t = 2)
f_names[0,1] = "Fig.3D_two_extremes"


fig, axs = plt.subplots(2,2, figsize=(6,6), sharex=True, sharey=True,
                        gridspec_kw=dict(wspace=0.05))

s = bs.coTrSplMechanistic()
s.set_param("rbp_e_down", 50)

for i in range(len(settings)):
    for j in range(len(settings[i])):
        ax = axs[i,j]
        vpols = vpol_profile_vpols
        s.set_raster(30001)
        s.set_runtime(1e4)
        s.params.update(settings[i,j])
        s.set_param("vpol", 50)
        
        if create_model_files:
            f_name = f_names[i,j] 
            if(not os.path.exists(parameters_table_dir)):
                os.makedirs(parameters_table_dir)
            print(f_name)
            s.toSBML(os.path.join(sbml_dir, f_name + ".xml"))
#            td_m.toSBML(os.path.join(sbml_dir, f_sbml))
            df, pars = s.get_parTimeTable()
            df_filtered = df[["from", "to"] + pars]
            df.to_csv(os.path.join(parameters_table_dir, f_name + ".csv"))
            df_filtered.to_csv(os.path.join(parameters_table_dir, f_name  + "_filtered.csv"))
        
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
            ax.plot(vpols, psis_no_rbp, lw=2, ls=":", label = "PSI: no RBP binding")
            ax.plot(vpols, psis_full_rbp, lw=2, ls=":", label = "PSI: 100% RBP binding")
            ax.plot(vpols, rets, lw=1, label="% retention isoform")
            ax.plot(vpols, rbp_reacts, lw=1.5, ls="--", label="% RBP bound pre-mRNA")
        
#        ax.set_xlabel("vpol [nt/s]")
#        ax.set_ylabel("PSI")
        ax.set_title("rbp pos: %d, rbp_br: %.2f" % (s.params["rbp_pos"], s.params["rbp_br_t"]),
                     fontsize = 11)
        ax.set_xscale("log")
if legend:
    axs[1,0].legend(bbox_to_anchor=(0., -0.50, 2., -0.3), loc='lower left',
           ncol=2)
axs[1,0].set_xlabel("vpol [nt/s]")
axs[1,1].set_xlabel("vpol [nt/s]")
axs[0,0].set_ylabel("PSI")
axs[1,0].set_ylabel("PSI")
fig.tight_layout()
#ax_leg = plt.subplot2grid(shape = (3,2), loc = (2,0), rowspan=1, colspan=2, fig=fig )
fig, ax = plt.subplots(figsize=(6,1))
ax.axis("off")
ax.legend(handles =axs[0,0].get_children()[0:5], ncol=2, fontsize=12)
fig.tight_layout()

# PARAMETER TIME COURSEs
if parameters_plot:
    s.set_runtime(20)
    pars = dict(vpol=50,
                rbp_pos = 480,
                rbp_br_t=1,
                rbp_e_down = 50,
                k1_t = 0.2,
                k2_t = 0.3,
                k3_t = 0.4,
                ret_t = 1e-3 )
    s.params.update(pars)
    if create_model_files:
        f_name = "Fig.3C_parameters"
        s.toSBML(os.path.join(sbml_dir, f_name + ".xml"))
        df, pars = s.get_parTimeTable()
        df_filtered = df[["from", "to"] + pars]
        df.to_csv(os.path.join(parameters_table_dir, f_name + ".csv"))
        df_filtered.to_csv(os.path.join(parameters_table_dir, f_name  + "_filtered.csv"))
    pars = s._evaluate_pars()
    rbp_pos = pars["rbp_pos"]
    vpol = pars["vpol"]
    rbp_e_up = pars["rbp_e_up"]
    rbp_e_down = pars["rbp_e_down"]
    rbp_h_c = pars["rbp_h_c"]
    
    
    fig, ax = plt.subplots(figsize=(7,2.3))
    ax = s.plot_parameters(parnames=["k1","k2","k3"], annotate=False, ax = ax, lw=3)
    ax = s.plot_parameters(parnames=["k1_inh","k2_inh","k3_inh"], annotate=False,
                           ax=ax, lw=6, ls=(0,(1,3))
                           ,dash_capstyle="round"
                           ,dash_joinstyle = "round"
            )
    ax.set_title("vpol: %d, rbp_pos: %d" % (vpol, rbp_pos))
    ax.set_title("vpol: %d, rbp_pos: %d, rbp_range:(%d, %d)" % (vpol, rbp_pos, rbp_e_up, rbp_e_down ),)
    ax.set_ylabel("Exon def. rates")
    ax.set_xlabel("time")
    ax.legend(fontsize=9)
    fig.tight_layout()
    
    fig, ax = plt.subplots(figsize=(7,1.8))
    ax = s.plot_parameters(parnames=["rbp_br"], annotate=False, ax=ax, lw=3)
    ax.set_ylabel("RBP binding rate")
    ax.set_xlabel("time")
    ax_twin = ax.twinx()
    ax_twin = s.plot_parameters(parnames=["k_ret"], annotate=False, ax=ax_twin, lw=3, c="orange")
    ax_twin.legend(loc="upper right")
    ax_twin.set_ylabel("Intron ret. rate")
    fig.tight_layout()


if(inhibition_plot):
    fig, ax = plt.subplots(figsize=(3.5,2.4))
    rbpp = rbp_pos
    u1_2_pos = pars["u1_ex2"]
    u2_2_pos = pars["u2_ex3"]
    rbp_poss = np.linspace(rbp_pos-rbp_e_up*2, rbp_pos+rbp_e_down*2,100)
    asym_pr = s.get_function("inhFunc")["lambda_f"]
    inh_curve_u11 = [asym_pr(rbp_p- rbpp, rbp_e_up, rbp_e_down, rbp_h_c) for rbp_p in rbp_poss]
    ax.plot(rbp_poss,inh_curve_u11, label = "$InhFunc$", linestyle="-", lw=3)
    ax.axvline(rbpp, ls = "-.")
    ax.axvline(u1_2_pos, ls = ":")
    ax.axvline(u2_2_pos, ls = ":")
#    ax.legend()
    
    ax.set_ylabel("Inh. strength")
    fig.tight_layout()

