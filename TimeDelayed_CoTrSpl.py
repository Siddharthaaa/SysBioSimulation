#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:37:19 2019

@author: timur
"""

import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np

sims_count = 10

ki0 = 1
ks0 = 0
ks1 = 1
skip_pos = 50 #nt 
#end_pos = 1000

runtime = 1e3

tau1 = 1

polvs = np.logspace(-1,3,101)
polvs = np.linspace(1e-3, 300, 101)

s1 = bs.SimParam("TD_CoTrSpl_unbranched",
                 runtime, 10001,
                 dict(ki = ki0, ks =ks0,
                      ks1 = ks1),
                 dict(preRNA = 1e3, Incl = 0, Skip = 0))

s1.add_reaction("preRNA*ki", {"Incl":1, "preRNA":-1})
s1.add_reaction("preRNA*ks", {"Skip":1, "preRNA":-1})

te = bs.TimeEvent(tau1, "ks=ks1")
s1.add_timeEvent(te)
s1.simulate()

for i in range(sims_count):
    psis = []
    for v0 in polvs:        
    #    tau1 = tau
        tau1 = skip_pos/v0
        te.set_time(tau1)
        s1.simulate()
        psi = s1.get_psi_mean(ignore_fraction=0.5)
#        print("PSI: ", psi)
        psis.append(psi)
    plt.plot(polvs, psis, c="blue", lw=0.2)
#plt.set_xlabel("v_elong")
plt.xlabel("v_elong")
plt.ylabel("PSI")




#################### Two time Events #######################################


ki0 = 1e-1
ks0 = 5e-1
ks1 = 1e-3
ks2 = 5e-1

tau1 = 2
tau2 = 4

nt_pos1 = 15
nt_pos2 = 150

runtime = 1000

polvs = np.linspace(1e-3, 1000, 101)
polvs = np.logspace(0,3,101)

s1 = bs.SimParam("TD_CoTrSpl_branched",
                 runtime, 10001,
                 dict(ki = ki0, ks =ks0,
                      ks1 = ks1, ks2 = ks2),
                 dict(preRNA = 1e3, Incl = 0, Skip = 0))

s1.add_reaction("preRNA*ki", {"Incl":1, "preRNA":-1})
s1.add_reaction("preRNA*ks", {"Skip":1, "preRNA":-1})

te1 = bs.TimeEvent(tau1, "ks=ks1")
te2 = bs.TimeEvent(tau2, "ks=ks2")
s1.add_timeEvent(te1)
s1.add_timeEvent(te2)


s1.simulate()
#s1.show_interface()



for i in range(sims_count):
    psis = []
    for v0 in polvs:        
    #    tau1 = tau
        tau1 = nt_pos1/v0
        tau2 = nt_pos2/v0
        te1.set_time(tau1)
        te2.set_time(tau2)
        
        s1.simulate()
        psi = s1.get_psi_mean(ignore_fraction=0.5)
#        print("PSI: ", psi)
        psis.append(psi)
    plt.plot(polvs, psis, c="blue", lw=0.2)
#plt.set_xlabel("v_elong")
plt.xlabel("v_elong")
plt.ylabel("PSI")
plt.xscale("log")
