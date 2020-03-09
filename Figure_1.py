#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:23:34 2020

@author: timur
"""


import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np

#settings 
steps_multiplier = 2

l = 8
l_many = l * steps_multiplier
n = 2
n_many = n * steps_multiplier

ki = 0.05
ks = 0.5
kesc = 0
tr_len = 300

res_type = "ODE"

l1 = l-n

vpol_50p = -ki * tr_len/(l*np.log(0.5))*l1

psi_slow = 1
psi_fast = ki/(ki+ks)

vpols = np.logspace(0,3,100)

fontsize=12
legend_fs = 9
legend_outside = False
plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize)
plt.rc("font", size=fontsize)

#settings  end

if legend_outside:
    figsize=(6.5, 5.2)
    leg_loc = (1.15, 0.3)
else:
    figsize=(3.7, 5.2)
    leg_loc = "best"

models = bs.coTrSplCommitment(50, tr_len=tr_len, l=l, m1=0, m2=0, k=0, n=n,
                    ki=ki, ks=ks, kesc=kesc, kesc_r=0)

step_m = models["step_m"]
td_m = models["td_m"]
psi_f = models["psi_analytic_f"]

models = bs.coTrSplCommitment(50, tr_len=tr_len, l=l_many, m1=0, m2=0, k=0, n=n_many,
                   ki=ki, ks=ks, kesc=kesc, kesc_r=0)
many_step_m = models["step_m"]

fig, axs = plt.subplots(2,1, figsize=figsize)
axs[0].set_xscale("log")
axs[1].set_xscale("log")


psis_step = []
psis_many_step = []
psis_td = []
psis_analyt = []

for vpol in vpols:
    step_m.set_param("vpol", vpol)
    step_m.simulate(ODE=True)
    psis_step.append(step_m.get_psi_end(res_type="ODE"))
    
    td_m.set_param("vpol", vpol)
    td_m.simulate(ODE=True)
    psis_td.append(td_m.get_psi_end(res_type="ODE"))
    
    many_step_m.set_param("vpol", vpol)
    many_step_m.simulate(ODE=True)
    psis_many_step.append(many_step_m.get_psi_end(res_type="ODE"))
#    psis_analyt.append(psi_f(vpol))

vpols_analyt = vpols[0::10]
psis_analyt = [psi_f(vpol2) for vpol2 in vpols_analyt]

axs[0].plot(vpols, psis_td, lw = 4, c = "green", label= "time delay model")
#axs[0].plot(vpols, psis_td, lw = 4, c = "green")
axs[0].plot(vpols_analyt, psis_analyt, "bo",ms=8,  label="analytic solution")
#axs[0].set_xlabel("vpol")
axs[0].set_ylabel("PSI")
axs[0].axvline(vpol_50p, ls="-", lw=1, color="blue", label = "50% commitment\nbefore $\\tau$ (P1-P6)")
axs[0].axhline(psi_slow, ls="-.", lw=1, color="green", label = "PSI (P1-P6)")
axs[0].axhline(psi_fast, ls="-.", lw=1, color="red", label = "PSI (P7-P8)")
#axs[0].legend(loc = (1.15,0.3), fontsize=legend_fs)
axs[0].legend(loc = leg_loc, fontsize=legend_fs)


axs[1].plot(vpols, psis_step, lw = 2, label = "few steps (%d)" % l)
axs[1].plot(vpols, psis_many_step, lw = 2, label = "many steps (%d)" % l_many)
axs[1].plot(vpols[0::10], psis_td[0::10], "bo", ms=8,
   label = "time delay model")
axs[1].set_xlabel("vpol")
axs[1].set_ylabel("PSI")
#axs[1].set_title("Multi-step models")

axs[1].legend(loc = leg_loc, fontsize=legend_fs)

[fig.tight_layout() for i in  range(3)]

td_m.set_param("vpol", 50)
td_m.set_runtime(6)
fig, ax = plt.subplots( figsize=(5,2))
ax = td_m.plot_parameters(parnames = "ki", ax=ax, annotate=False, c="green", lw=3, ls="--")
ax = td_m.plot_parameters(parnames = "ks", ax=ax, annotate=False, c="red", lw=3, ls="--")
ax.set_xlabel("time")
#ax.set_ylabel("value")
ax.set_title("Parameters (vpol=%.1f)" % td_m.params["vpol"])
[fig.tight_layout() for i in  range(3)]
