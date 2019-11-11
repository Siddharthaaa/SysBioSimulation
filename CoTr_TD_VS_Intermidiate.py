# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:13:21 2019

@author: Timur
"""



import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np

sims_count = 100

gene_length = 1000 #nt
init_mol_count = 1000
k=2
l=2
m = 2
n = 3
chain_len = k+l+m+n

vpol = 100. # nt/s
ki = 5e-2
ks = 1e-2
kesc = 0.3
#kesc = 10
runtime = 1000
k_elongs = np.logspace(1,3,31)

avg_tr_time = gene_length/vpol
kelong = (k+l+m+n)/avg_tr_time
s1 = bs.SimParam("CoTrSpl_general",
                 runtime, 10001,
                 dict(k_elong=kelong,  ki = ki, ks = ks, kesc = kesc),
                 dict(p1=init_mol_count, Incl = 0, Skip = 0))

#s1.add_reaction("vsyn", {"p1":1})
for i in range(1, k+l+n+m):
    p1 = "p" + str(i)
    p2 = "p" + str(i+1)
#    e1 = "e" + str(i)
#    e2 = "e" + str(i+1)
    s1.add_reaction("k_elong*" + p1, {p1:-1, p2:1} )
    s1.add_reaction("ki *" + p1, {p1:-1, "Incl":1})
#    s1.add_reaction("k_elong*" + e1, {e1:-1, e2:1})
s1.add_reaction("ki *" + p2, {p2:-1, "Incl":1})

for i in range(k+1, k+l+m+n):
    e1 = "e" + str(i)
    e2 = "e" + str(i+1)
    s1.add_reaction("k_elong*" + e1, {e1:-1, e2:1})

for i in range(k, k+l):
    p = "p" + str(i+1)
    e = "e" + str(i+1)
    s1.add_reaction("kesc*" + p, {p:-1, e:1})

for i in range(k+l+m, k+l+m+n):
    p = "p" + str(i+1)
    e = "e" + str(i+1)
    s1.add_reaction("ks*" + p, {p:-1, "Skip":1})
    s1.add_reaction("ks*" + e, {e:-1, "Skip":1})
    
#s1.add_reaction("d*Skip", {"Skip":-1})
#s1.add_reaction("d*Incl", {"Incl":-1})

s1.compile_system()
#s1.draw_pn(engine="dot", rates=False)
step_sim = s1


s1 = bs.SimParam("CoTrSpl_general_TD",
                 runtime, 10001,
                 dict(ki = ki, ks = 0, ks1 = ks, kesc = 0, kesc1=kesc, d=0.04),
                 dict(mRNA = init_mol_count, Incl = 0, Skip = 0))

s1.add_reaction("mRNA*ki", {"Incl":1, "mRNA":-1})
s1.add_reaction("mRNA*ks", {"Skip":1, "mRNA":-1})
s1.add_reaction("mRNA*kesc", {"mRNAbr":1, "mRNA":-1})
s1.add_reaction("mRNAbr*ks", {"Skip":1, "mRNAbr":-1})

tau1 = avg_tr_time/chain_len*k
tau2 = avg_tr_time/chain_len*(k+l)
tau3 = avg_tr_time/chain_len*(k+l+m)
te1 = bs.TimeEvent(tau1, "kesc=kesc1")
te2 = bs.TimeEvent(tau2, "kesc=0")
te3 = bs.TimeEvent(tau3, "ks=ks1")
s1.add_timeEvent(te1)
s1.add_timeEvent(te2)
s1.add_timeEvent(te3)

s1.draw_pn(engine="dot", rates=False)
td_sim = s1
#s1.show_interface()
# Plot many realizations
ax = None
fig, ax = plt.subplots()
psis_all1 = np.zeros((int(sims_count), len(k_elongs)))
psis_all2 = np.zeros((int(sims_count), len(k_elongs)))

for i in range(sims_count):
    for j, vpol in enumerate(k_elongs):
        avg_tr_time = gene_length/vpol
        kelong = (k+l+m+n)/avg_tr_time
        step_sim.set_param("k_elong", kelong)
        step_sim.simulate()
        psis_all1[i,j] = step_sim.get_psi_mean()
        
        
        tau1 = avg_tr_time/chain_len*k
        tau2 = avg_tr_time/chain_len*(k+l)
        tau3 = avg_tr_time/chain_len*(k+l+m)
        te1.set_time(tau1)
        te2.set_time(tau2)
        te3.set_time(tau3)
        td_sim.simulate()
        psis_all2[i,j] = td_sim.get_psi_mean()

ax.boxplot(psis_all1, labels = k_elongs)
ax.boxplot(psis_all2, labels = k_elongs)
        
#ax.set_xscale("log")
ax.axvline(ki, linestyle="--", color="green", label="k_elong=ki")
ax.axvline(kesc, linestyle="--", color="red", label="k_elong=kesc")
ax.legend()
ax.set_title("Late strong inh.")
ax.set_xlabel("kelong")
ax.set_ylabel("PSI")
