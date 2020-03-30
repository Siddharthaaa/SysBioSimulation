#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:31:39 2020

@author: timur
"""


import os
import bioch_sim as bs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np
import sympy as sy


parameters_table_dir = os.path.join("docs", "pars_csv")
sbml_dir = os.path.join("docs", "sbml")
create_model_files = True

skewed_RBP_effect = True
plot_psi_landscape = False
inhibition_plot_inside = False

sim_series_rbp_pos = True
sim_series_rbp_pos_vpol = 50
if skewed_RBP_effect:
    rbp_upstream_radius = 1
    rbp_downstream_radius = 80
    rbp_inh_title = "Right-skewed RBP effect"
    f_name = "Fig.4_skewed"
else:
    rbp_upstream_radius = 30
    rbp_downstream_radius = 40
    rbp_inh_title = "Bidirectional  RBP effect"
    f_name = "Fig.4_bidirectional"
sim_series_rbp_pos_add_vpols = [10, 50, 200, 10000]
rbp_posistions = np.linspace(100, 600, 101)

figsize=(8,4.5)



pars = dict(vpol=50,
            rbp_pos = 480,
            rbp_br_t=5,
            rbp_inh=0.99,
            k1_t = 1,
            k2_t = 0.1,
            k3_t = 1,
            ret_t = 1e-3,
            rbp_e_up = rbp_upstream_radius,
            rbp_e_down = rbp_downstream_radius
            )


s = bs.coTrSplMechanistic()
s.params.update(pars)

if create_model_files:
#    f_name = "Fig.4_parameters"
    s.toSBML(os.path.join(sbml_dir, f_name + ".xml"))
    df, pars = s.get_parTimeTable()
    df_filtered = df[["from", "to"] + pars]
    df.to_csv(os.path.join(parameters_table_dir, f_name + ".csv"))
    df_filtered.to_csv(os.path.join(parameters_table_dir, f_name  + "_filtered.csv"))

s.compile()
pars = s._evaluate_pars()
rbp_pos = pars["rbp_pos"]
vpol = pars["vpol"]
rbp_e_up = pars["rbp_e_up"]
rbp_e_down = pars["rbp_e_down"]
u1_1_pos = pars["u1_ex1"]
u2_1_pos = pars["u2_ex2"]
u1_2_pos = pars["u1_ex2"]
u2_2_pos = pars["u2_ex3"]
rbp_inh = pars["rbp_inh"]



if sim_series_rbp_pos:
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=1, fig=fig)
    ax1_leg = plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=1, fig=fig)
    ax1_leg.axis("off")
    leg1_handels = []
    
    ax2 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=1, fig=fig, sharex=ax1)
    ax2_leg = plt.subplot2grid((3,3), (1,2), colspan=1, rowspan=1, fig=fig)
    if not inhibition_plot_inside:
        ax2_leg.axis("off")
#    ax2.set_xticklabels([])
    
    ax3 = plt.subplot2grid((3,3), (2,0), colspan=2, rowspan=1, fig=fig, sharex=ax1)
    ax3_leg = plt.subplot2grid((3,3), (2,2), colspan=1, rowspan=1, fig=fig)
    ax3_leg.axis("off")
    ax3.set_xlabel("RBP position")
    
    s.set_runtime(1e4)
    s.set_raster(30001)
    s.set_param("vpol", sim_series_rbp_pos_vpol)
    psis = []
    rets = []
    k1 = []
    k2 = []
    k3 = []
    
    rbp_br = s.params["rbp_br_t"]
    s.set_param("rbp_br_t", 0)
    s.simulate(stoch=False, ODE=True)
    incl = s.get_res_col("Incl", method="ODE")[-1]
    skip = s.get_res_col("Skip", method="ODE")[-1]
    psi_default = incl/(incl+skip)
    s.set_param("rbp_br_t", rbp_br)
    for i, rbp_pos in enumerate(rbp_posistions):
        print("Sim count: %d" % i)
        s.set_param("rbp_pos", rbp_pos)
        params = s._evaluate_pars()
        k1.append(params["k1_inh_t"])
        k2.append(params["k2_inh_t"])
        k3.append(params["k3_inh_t"])
        s.simulate(stoch=False, ODE=True)
        incl = s.get_res_col("Incl", method="ODE")[-1]
        skip = s.get_res_col("Skip", method="ODE")[-1]
        rbp_r = s.get_res_col("SkipInh", method="ODE")[-1]
        rbp_r += s.get_res_col("InclInh", method="ODE")[-1]
#        psi = s.get_psi_mean(ignore_fraction = 0.4)
        psi = incl/(incl+skip)
        psis.append(psi)   
        ret = s.get_res_col("ret", "ODE")[-1]
        rets.append(ret/s.init_state["P000"])
    
    
#    fig, ax = plt.subplots(figsize=(6,2))
    ax=ax1
    leg1_handels.append(ax.plot(rbp_posistions, psis, lw=4, label = "PSI")[0])
    leg1_handels.append(ax.axhline(psi_default, lw=2, c ="black", ls="--",
                                   label="PSI default"))
    leg1_handels.append(ax.plot(rbp_posistions, rets, lw=1, label="% retention")[0])
   
#    ax.set_xlabel("RBP position")
    ax.set_ylabel("PSI")
#    ax.set_title("u11: %d; u21: %d; u12: %d; u22: %d; Radius:(%d, %d)" % 
#                 (u1_1_pos, u2_1_pos, u1_2_pos, u2_2_pos, rbp_e_up, rbp_e_down))
#    ax.set_title("vpol: %d, max. inh.: %.2f; RBP radius:(%d, %d)" % 
#                 (sim_series_rbp_pos_vpol, rbp_inh, rbp_e_up, rbp_e_down))
    ax.set_title("RBP postion-PSI profiles")
    ax.axvline(u1_1_pos, linestyle="-.",lw =0.7)#, color = "red")
    ax.axvline(u2_1_pos, linestyle="-.",lw =0.7)#, color = "red")
    ax.axvline(u1_2_pos, linestyle="-.", lw =0.7)#, color = "red")
    leg1_handels.append(ax.axvline(u2_2_pos, label="Splice sites", linestyle="-.", lw =0.7))#, color = "red")
    ax.axhspan(psi_default, np.max(psis), facecolor = "green", alpha = 0.1)
    ax.axhspan(psi_default, 0, facecolor = "red", alpha = 0.1)
    ax1_leg.legend(handles = leg1_handels, loc = "lower left")
    #additional vpols
    for vp in sim_series_rbp_pos_add_vpols:
        s.simulate_ODE = True
        s.set_param("vpol", vp)
        ax3 = s.plot_par_var_1d(par="rbp_pos", vals=rbp_posistions,
                               label = "vpol: " + str(vp), ax = ax3,
                               plot_args=dict(lw=2, ls="-"),
                               func=s.get_psi_end, res_type="ODE")
    
    ax3_leg.legend(handles = ax3.get_children()[0:4], loc = "lower left")
    ax3.set_ylabel("PSI")
    
#    ax2 = ax.twinx()
#    fig, ax2 = plt.subplots(figsize=(6,1.5))
    alpha = 1
    ax2.plot(rbp_posistions, k1, lw =2, c = "blue", alpha=alpha, ls = "-.", label="k1_inh")
    ax2.plot(rbp_posistions, k2, lw =2, c = "red", alpha=alpha, ls = "-.", label="k2_inh")
    ax2.plot(rbp_posistions, k3, lw =2, c = "green", alpha=alpha, ls = "-.", label = "k3_inh")
#    ax2.set_xlabel("RBP position")
    ax2.set_ylabel("ki_inh")
    
    ax1.tick_params(labelbottom=False)    
    ax2.tick_params(labelbottom=False)    
    fig.tight_layout()
    if not inhibition_plot_inside:
        ax2_leg.legend(handles = ax2.get_children()[0:3], loc = "lower left")
    else:
        ax2.legend(loc = "best")
    
if plot_psi_landscape:

    s.set_runtime(1e5)
    s.set_raster(100001)
    vpols = np.linspace(1,400,100)
    vpols = np.logspace(0,4,51)
    rbp_poss = np.linspace(430, 455, 41)
    rbp_poss = rbp_posistions
    X, Y = np.meshgrid(rbp_poss, np.log10(vpols))
    
    Z = np.zeros((len(vpols), len(rbp_poss)))
    rets = np.array(Z)
    
    for i, vpol in enumerate(vpols):
        s.set_param("vpol", vpol)
        for j, rbp_p in enumerate(rbp_poss):
            s.set_param("rbp_pos", rbp_p)
#            print(vpol, rbp_p)
            
            res = s.simulate(stoch=False, ODE=True)
            incl = s.get_res_col("Incl", method="ODE")[-1]
            skip = s.get_res_col("Skip", method="ODE")[-1]
            ret = s.get_res_col("ret", method="ODE")[-1]
            if(ret<0 or incl <0 or skip <0):
                print(ret, incl, skip)
            psi = incl/(incl+skip)
            ret_perc = ret/s.init_state["P000"]
            Z[i,j] = psi
            rets[i,j] = ret_perc
            
            
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z,  cmap=cm.coolwarm)
    ax.set_xlabel("RBP position")
    ax.set_ylabel("log10 (vpol [nt/s])")    
    ax.set_zlabel("PSI")
#    ax.set_yscale("log")
#    ax.yaxis._set_scale('log')
    
    #heatmaps 
    step = 5
    indx_x = np.arange(0, len(rbp_poss)-1, step)
    indx_y = np.arange(0, len(vpols), step)
    
    fig, ax  = plt.subplots()
    im = ax.imshow(rets)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("ret %", rotation=-90, va="bottom")
    ax.set_xticks(indx_x)
    ax.set_yticks(indx_y)
    # ... and label them with the respective list entries.
    col_labels = [ "%.0f" % v for v in rbp_poss[indx_x]]
    row_labels = ["%.2f" % v for v  in vpols[indx_y]]
    ax.set_xticklabels(col_labels, fontsize = 10, rotation=60)
    ax.set_yticklabels(row_labels, fontsize = 10)
    ax.set_xlabel("RBP position")
    ax.set_ylabel("vpol")
    
    fig, ax  = plt.subplots(figsize=(8,4))
    im = ax.imshow(Z)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("PSI", rotation=-90, va="bottom")
    ax.set_xticks(indx_x)
    ax.set_yticks(indx_y)
    pars = s._evaluate_pars()
    ax.set_title("k1: %.2f, k2: %.2f, k3: %.2f, rbp_br: %.2f, max. inh: %.3f" % 
                 (pars["k1_t"], pars["k2_t"], pars["k3_t"], pars["rbp_br_t"], pars["rbp_inh"]))
    ax.set_title("PSI Landscape")
    # ... and label them with the respective list entries.
#    col_labels = rbp_poss[indx_x]
#    row_labels = vpols[indx_y]
    ax.set_xticklabels(col_labels, fontsize = 10, rotation=60)
    ax.set_yticklabels(row_labels, fontsize = 10)
    ax.set_xlabel("RBP position")
    ax.set_ylabel("vpol")

if(inhibition_plot_inside == False):
    fig, ax = plt.subplots(figsize=(2.5, 2))
else:
    ax = ax2_leg
rbpp = 0
u1_2_pos = pars["u1_ex2"]
u2_2_pos = pars["u2_ex3"]
rbp_h_c = pars["rbp_h_c"]
rbp_poss = np.linspace(-rbp_e_up*2, rbp_e_down*2,100)
inhFunc = s.get_function("inhFunc")["lambda_f"]
inh_curve_u11 = [inhFunc(rbp_p- rbpp, rbp_e_up, rbp_e_down, rbp_h_c) for rbp_p in rbp_poss]
ax.plot(rbp_poss,inh_curve_u11,
#            label = "$InhFunc$",
        linestyle="-", lw=3)
ax.axvline(rbpp, ls = "-.")
#    ax.axvline(u1_2_pos, ls = ":")
#    ax.axvline(u2_2_pos, ls = ":")
#    ax.legend()
ax.set_title("RBP inhibition effect")
ax.set_title(rbp_inh_title, fontsize=10)
ax.set_ylabel("inh. strength")
ax.set_xlabel("Distance in [nt]")
fig.tight_layout()