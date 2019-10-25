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
                50, 10001, 
                {"k1":0, "k2":0, "k3":0,"d":1, "spl_r":1, "tau1":tau1, "tau2":tau2, "tau3":tau3},
                {"S000":100000, "S001":0, "S010":0, "S011":0, "S100":0,
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

s.add_reaction("spl_r*S011", {"S011":-1, "SecIR":1})
s.add_reaction("spl_r*S110", {"S110":-1, "FirstIR":1})
s.add_reaction("spl_r*S101", {"S101":-1, "Skip":1})
s.add_reaction("spl_r*S111", {"S111":-1, "Incl":1})

s.add_reaction("spl_r*S000", {"S000":-1, "FullIR":1})
s.add_reaction("spl_r*S001", {"S001":-1, "FullIR":1})
s.add_reaction("spl_r*S010", {"S010":-1, "FullIR":1})
s.add_reaction("spl_r*S100", {"S100":-1, "FullIR":1})

s.add_timeEvent(bs.TimeEvent(tau1, "k1=2"))
s.add_timeEvent(bs.TimeEvent(tau2, "k2=2"))
s.add_timeEvent(bs.TimeEvent(tau3, "k3=2"))
s.simulate_ODE = True
s.simulate()

#s.show_interface()

#s.add_reaction("d*S011", {"S011":-1})
#s.add_reaction("d*S110", {"S110":-1})
#s.add_reaction("d*S101", {"S101":-1})
#s.add_reaction("d*S111", {"S111":-1})

taus = np.linspace(0,2,101)
psis = []
for tau in taus:
        
    tau1 = tau
    tau2 = tau1*445/212;   
    tau3 = tau1*692/212;
    s.delete_timeEvents()    
    s.add_timeEvent(bs.TimeEvent(tau1, "k1=2"))
    s.add_timeEvent(bs.TimeEvent(tau2, "k2=2"))
    s.add_timeEvent(bs.TimeEvent(tau3, "k3=2"))
    s.simulate()
    psi = s.get_psi_mean(ignore_fraction=0.5)
    print("PSI: ", psi)
    psis.append(psi)
    
    
plt.plot(taus, psis)
s.draw_pn(engine="dot", rates = False)
