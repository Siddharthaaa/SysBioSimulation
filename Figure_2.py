# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:42:12 2020

@author: Timur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:23:34 2020

@author: timur
"""



import bioch_sim as bs
import matplotlib.pyplot as plt

import numpy as np

#settings 
fontsize=11
legend_fs = 9
legend_outside = True
set_title = False
esc_share = True
plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize)
plt.rc("font", size=fontsize)

l = 8
#n = 2
vpol = 50
tr_len = 300
vpols = np.logspace(0,3,100)


models = []
# MODEL 2: S-Shape
topol = dict(l=8, m1=0, m2=1, k=0, n=2,
                    ki= 0.1, ks=0.02, kesc=0.5)
 #verticals
vs = {}
vpol_50p = -(topol["kesc"]+topol["ki"]) * tr_len/(l*np.log(0.5))*topol["m2"]
vs["50% commitment\nbefore $\\tau_{inh,2}$ (P1)"] = vpol_50p
#vpol_50p = -topol["kesc"] * tr_len/(l*np.log(0.5))*topol["m1"]
#vs["50% commitment\nbetween $\\tau_{inh,1}$ and $\\tau1_{inh,2}$\n (P1-P2)"] = vpol_50p
#    horizintals
hs= {}
hs["PSI (P1)"] = topol["ki"]/(topol["ki"] + topol["kesc"])
hs["PSI (P7-P8)"] = topol["ki"]/(topol["ki"] + topol["ks"])
models.append(
        dict(name = "early inh. S-Shape",
             topol=topol, vs=vs, hs=hs))

#MODEL 3: Bell-Shape
topol = dict(l=8, m1=0, m2=1, k=0, n=2,
                    ki= 1e-1, ks=2e-1, kesc=0.2)
 #verticals
vs = {}
vpol_50p = -topol["ki"] * tr_len/(l*np.log(0.5))*topol["m1"]
vs["50% commitment\nbefore $\\tau_{inh,2}$ (P1)"] = vpol_50p
vpol_50p = -(topol["kesc"]+topol["ki"]) * tr_len/(l*np.log(0.5))*topol["m2"]
vs["50% commitment\nbetween $\\tau_{inh,1}$ and $\\tau_{inh,2}$\n (P1-P2)"] = vpol_50p
#    horizintals
hs= {}
hs["PSI (P1)"] = topol["ki"]/(topol["ki"] + topol["kesc"])
hs["PSI (P7-P8)"] = topol["ki"]/(topol["ki"] + topol["ks"])
#hs["PSI (P2-P6)"] = 1
models.append(
        dict(name = "early inh. Bell-shape",
             topol=topol, vs=vs, hs=hs))


#MODEL 1: U-Shape
topol = dict(l=8, m1=2, m2=3, k=0, n=2,
                    ki= 5e-2, ks=5e-3, kesc=0.5)
 #verticals
vs = {}
vpol_50p = -topol["ki"] * tr_len/(l*np.log(0.5))*topol["m1"]
vs["50% commitment\nbefore $\\tau_{inh,1}$ (P1-P2)"] = vpol_50p
vpol_50p = -(topol["kesc"]+topol["ki"]) * tr_len/(l*np.log(0.5))*topol["m2"]
vs["50% commitment\nbetween $\\tau_{inh,1}$ and $\\tau_{inh,2}$\n (P1-P2)"] = vpol_50p
#    horizintals
hs= {}
hs["PSI (P1-P2)"] = 1
hs["PSI (P7-P8)"] = topol["ki"]/(topol["ki"] + topol["ks"])
hs["PSI (P3-P5)"] = topol["ki"]/(topol["ki"] + topol["kesc"])
models.append(
        dict(name = "late inh. U-shape",
             topol=topol, vs=vs, hs=hs))

m_c = len(models)
h = 2.7


if legend_outside:
    figsize=(5.4, h*m_c)
    leg_loc = (1.10 + esc_share*0.15, 0.05 + esc_share*0.2)
    leg_loc_share = (1.10 + esc_share*0.15, 0.05)
    cols = 1
else:
    figsize=(3, h*m_c)
    leg_loc = "best"
    cols = 1
    
fig_all, axs = plt.subplots(len(models), cols, figsize=figsize,
                        squeeze=False,
                        sharex=True)

for m_i, model in enumerate(models):
    step_m, td_m, psi_f = bs.coTrSplCommitment(vpol=50, tr_len=300,
                                       **model["topol"]).values()
    step_m.set_runtime(1e4)
    td_m.set_runtime(1e4)
    res_type = "ODE"
   
    axs[m_i,0].set_xscale("log")
#    axs[1].set_xscale("log")
    
    
    psis_step = []
    psis_td = []
    psis_analyt = []
    esc_det_td = []
    
    for vpol in vpols:
        step_m.set_param("vpol", vpol)
        step_m.simulate(ODE=True)
        psis_step.append(step_m.get_psi_end(res_type=res_type))
        
        td_m.set_param("vpol", vpol)
        td_m.simulate(ODE=True)
        psis_td.append(td_m.get_psi_end(res_type=res_type))
        esc = td_m.get_res_col("ESC", method=res_type)[-1]
        esc_det_td.append(esc/td_m.init_state["mRNA"])
        
       
    
    vpols_analyt = vpols[0::10]
    psis_analyt = [psi_f(vpol2) for vpol2 in vpols_analyt]
    
    axs[m_i,0].plot(vpols, psis_td, lw = 4, c = "green", label= "Time Delay model")
    axs[m_i,0].plot(vpols, psis_step, lw = 1, c = "red", label = "step model")
    axs[m_i,0].plot(vpols_analyt, psis_analyt, "bo",ms=8,  label="analytic")
    
    if(esc_share):
        ax_twin = axs[m_i,0].twinx()
        ax_twin.plot(vpols, esc_det_td, ls="--", c="black", lw=1, label="share of $mRNA_{inh}$")
        ax_twin.legend(loc = leg_loc_share, fontsize=legend_fs)
    
    axs[m_i,0].set_ylabel("PSI")
    if(set_title == True):
        axs[m_i,0].set_title(model["name"])
    
    for i, (k, v)  in enumerate(model["vs"].items()):
        axs[m_i,0].axvline(v, ls="-", lw=1,c="C"+str(i), label = k)
    for i, (k, v)  in enumerate(model["hs"].items()):
        axs[m_i,0].axhline(v, ls=":", lw=2,c="C"+str(i), label = k)
    

#    axs[1].plot(vpols, psis_step, lw = 2, label = "step model (%d)" % l)
##    axs[1].plot(vpols, psis_many_step, lw = 2, label = "many steps (%d)" % l_many)
#    axs[1].plot(vpols[0::10], psis_td[0::10], "bo", ms=8,
#       label = "time delay model")
#    axs[1].set_xlabel("vpol")
#    axs[1].set_ylabel("PSI")
#    axs[1].set_title("Multi-step models")
    
    axs[m_i,0].legend(loc = leg_loc, fontsize=legend_fs)
    

    
    td_m.set_param("vpol", 50)
    td_m.set_runtime(6)
    fig, ax = plt.subplots(figsize=(5,2))
    ax = td_m.plot_parameters(parnames = "ki", ax=ax, annotate=False, c="green", lw=3, ls="--")
    ax = td_m.plot_parameters(parnames = "ks", ax=ax, annotate=False, c="red", lw=3, ls="--")
    ax = td_m.plot_parameters(parnames = "kesc", ax=ax, annotate=False, c="blue", lw=3, ls="--")
    ax.set_xlabel("time")
    #ax.set_ylabel("value")
    ax.set_title("Parameters (vpol=%.1f)" % td_m.params["vpol"])
    ax.legend(loc="best")
    [fig.tight_layout() for i in  range(3)]

axs[m_i,0].set_xlabel("vpol")
[fig_all.tight_layout() for i in  range(1)]