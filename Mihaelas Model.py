#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:25:36 2019

@author: timur
"""



import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np

tau1 = 2
tau2 = tau1*445/212;   
tau3 = tau1*692/212;


s = bs.SimParam("CoTrSpl delayd pars",
                10, 10001, 
                {"k1":0, "k2":0, "k3":0,"d":1, "ks":1e-1},
                {"S000":10000, "S001":0, "S010":0, "S011":0, "S100":0,
                           "S101":0, "S110":0, "S111":0})


s.add_reaction("k1*S000", {"S000":-1, "S001":1}, "1. Exon binding")
s.add_reaction("k2*S000", {"S000":-1, "S010":1}, "2. Exon binding")
s.add_reaction("k3*S000", {"S000":-1, "S100":1}, "3. Exon binding")
s.add_reaction("k2*S001", {"S001":-1, "S011":1}, "2. Exon binding")
s.add_reaction("k3*S001", {"S001":-1, "S101":1}, "3. Exon binding")
s.add_reaction("k1*S010", {"S010":-1, "S011":1}, "1. Exon binding")
s.add_reaction("k3*S010", {"S010":-1, "S110":1}, "3. Exon binding")
s.add_reaction("k1*S100", {"S100":-1, "S101":1}, "1. Exon binding")
s.add_reaction("k2*S100", {"S100":-1, "S110":1}, "2. Exon binding")
s.add_reaction("k3*S011", {"S011":-1, "S111":1}, "3. Exon binding")
s.add_reaction("k1*S110", {"S110":-1, "S111":1}, "1. Exon binding")
s.add_reaction("k2*S101", {"S101":-1, "S111":1}, "2. Exon binding")

s.add_reaction("ks*S011", {"S011":-1, "SecIR":1})
s.add_reaction("ks*S110", {"S110":-1, "FirstIR":1})
s.add_reaction("ks*S101", {"S101":-1, "Skip":1})
s.add_reaction("ks*S111", {"S111":-1, "Incl":1})

s.add_reaction("ks*S000", {"S000":-1, "FullIR":1})
s.add_reaction("ks*S001", {"S001":-1, "FullIR":1})
s.add_reaction("ks*S010", {"S010":-1, "FullIR":1})
s.add_reaction("ks*S100", {"S100":-1, "FullIR":1})

te1= bs.TimeEvent(tau1, "k1=5e-2")
te2= bs.TimeEvent(tau2, "k2=5e-2")
te3= bs.TimeEvent(tau3, "k3=5e-2")
s.add_timeEvent(te1)
s.add_timeEvent(te2)
s.add_timeEvent(te3)
s.simulate_ODE = True
s.simulate()
ax = s.plot_series(products=["S000","S001","S010","S100"])
s.show_interface()

s._set_color("S000", "red")
ax, ax2 = s.plot_course(res=["ODE"], products=["S000"], products2=["S001","S010","S100"])
ax.set_ylabel("")
ax.set_xlabel("$t$")
ax.axvline(tau1,0,1, color="green", linestyle="--", label="$\\tau1$")
ax.axvline(tau2,0,1, color="red", linestyle="--", label="$\\tau2$")
ax.axvline(tau3,0,1, color="blue", linestyle="--", label="$\\tau3$")
ax.legend(loc = 6)
ax2.legend(loc=7)

#s.add_reaction("d*S011", {"S011":-1})
#s.add_reaction("d*S110", {"S110":-1})
#s.add_reaction("d*S101", {"S101":-1})
#s.add_reaction("d*S111", {"S111":-1})

te1.set_action("k1=5")
te2.set_action("k2=0.05")
te3.set_action("k3=5")
s.compile_system()
elvs = np.linspace(1,100,51)
s.simulate_ODE=False
taus = np.linspace(0,2,51)
s.set_runtime(1e6)
#for tau in taus:
for i in range(10):
    psis = []
    for v0 in elvs:        
    #    tau1 = tau
        tau1 = 212/v0
        tau2 = tau1*445/212;   
        tau3 = tau1*692/212;
#        s.delete_timeEvents()
        te1.set_time(tau1)
        te2.set_time(tau2)
        te3.set_time(tau3)
        s.simulate()
        psi = s.get_psi_mean(ignore_fraction=0.5)
        print("PSI: ", psi)
        psis.append(psi)
    plt.plot(elvs, psis, c="blue", lw=0.2)
#plt.set_xlabel("v_elong")
plt.xlabel("v_elong")
plt.ylabel("PSI")
plt.title("20 stochastic realizations")
#s.draw_pn(engine="dot", rates = True)
