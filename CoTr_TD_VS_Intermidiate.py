# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:13:21 2019

@author: Timur
"""



import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np

sims_count = 10

gene_length = 800 #nt
init_mol_count = 1000

#select model:
# 1: unbranched few steps
# 2: unbranched many steps

# 3: branched late inh few steps
# 4: branched late inh many steps

# 5: branched early inh few steps
# 6: branched early inh many steps

# 7: branched early inh bell few steps
# 8: branched early inh bell many steps

model_id = 3

k_elongs = np.logspace(0,3.2,40)

kesc = 1

if model_id == 1:
    k = 6
    l = 0
    m = 0
    n = 2
    ki = 5e-2
    ks = 5e-1
    kesc = 0
#    
#    psi_slow = 1
#    psi_fast = ki/(ki+ks)
#    psi_inter = 0



if model_id == 2:
    k = 60
    l = 0
    m = 0
    n = 20
    ki = 5e-2
    ks = 5e-1

if model_id == 3:
    k=2
    l=3
    m = 1
    n = 2
    
    ki = 5e-2
    ks = 1e-3
    kesc = 0.5
    
#    psi_slow = 1
#    psi_fast = ki/(ki+ks)
#    psi_inter = ki/(ki+kesc)

if model_id == 4:
    k=20
    l=30
    m = 10
    n = 20
    
    ki = 5e-2
    ks = 1e-3
    kesc = 0.5
    

if model_id == 5:
    k=0
    l=1
    m = 4
    n = 3
    
    ki = 1e-1
    ks = 1e-2
    kesc = 0.5
    k_elongs = np.logspace(0,3,40)
    
#    psi_slow = ki/(ki+kesc)
#    psi_fast = ki/(ki+ks)
#    psi_inter = ki/(ki+kesc)

if model_id == 6:
    k=0
    l=10
    m = 40
    n = 30
    
    ki = 1e-1
    ks = 1e-2
    kesc = 0.5
    k_elongs = np.logspace(0,3,40)
    

if model_id == 7:
    k=0
    l=1
    m = 5
    n = 2
    
    ki = 1e-1
    ks = 2e-1
    kesc = 2e-1
    k_elongs = np.logspace(0,3.2,40)

if model_id == 8:
    k=0
    l=10
    m = 50
    n = 20
    
    ki = 1e-1
    ks = 2e-1
    kesc = 2e-1
    k_elongs = np.logspace(0,3,40)

if model_id == 0:
    k=1
    l=20
    m = 1
    n = 1
    
    ki = 1e-1
    ks = 1e-2
    kesc = 2e-1
    k_elongs = np.logspace(0,3.2,40)

def psi_analyticaly(vpol, gene_length, k, l , m, n, ki, ks, kesc):
    avg_tr_time = gene_length/vpol
    kelong = (k+l+m+n)/avg_tr_time
    
    A1 = ki*k/kelong
    pi1 = 1-np.exp(-A1)
    A23 = (ki+kesc)*l/kelong
    pis2 = (1-pi1)*(1-np.exp(-A23))
    pi2 = pis2*ki/(ki+kesc)
#    pi2 = 1-np.exp(-ki*l/kelong)
    A4 = ki*m/kelong
    pi3 = (1-pi1 - pis2) * (1-np.exp(-A4))
    pi4 = (1 - pi1 -pis2-pi3)*ki/(ki+ks)
    
    pi = pi1 + pi2 + pi3 + pi4
    
    return pi

klm = k+l+m
psi_inter =l*ki/(l*ki+ l*kesc)
#psi_inter =ki/(ki+kesc)
 
if(k == 0):
    psi_slow = ki/(ki+kesc)
else:
    psi_slow = ki/ki

psi_fast = ki/(ki+ks)

chain_len = k+l+m+n
vpol = 50. # nt/s
#kesc = 10
runtime = 10000


avg_tr_time = gene_length/vpol
kelong = (k+l+m+n)/avg_tr_time
s1 = bs.SimParam("CoTrSpl_general",
                 runtime, 10001,
                 dict(k_elong=kelong,  ki = ki, ks = ks, kesc = kesc, d=1),
                 dict(p1=init_mol_count, Incl = 0, Skip = 0))

#s1.add_reaction("vsyn", {"p1":1})
for i in range(1, chain_len):
    p1 = "p" + str(i)
    p2 = "p" + str(i+1)
#    e1 = "e" + str(i)
#    e2 = "e" + str(i+1)
    s1.add_reaction("k_elong*" + p1, {p1:-1, p2:1} )
    s1.add_reaction("ki *" + p1, {p1:-1, "Incl":1})
#    s1.add_reaction("k_elong*" + e1, {e1:-1, e2:1})
s1.add_reaction("ki *" + p2, {p2:-1, "Incl":1})

for i in range(k+1, chain_len):
    e1 = "e" + str(i)
    e2 = "e" + str(i+1)
    s1.add_reaction("k_elong*" + e1, {e1:-1, e2:1})

for i in range(k, k+l):
    p = "p" + str(i+1)
    e = "e" + str(i+1)
    s1.add_reaction("kesc*" + p, {p:-1, e:1})

for i in range(k+l+m, chain_len):
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

#s1.draw_pn(engine="dot", rates=False)
td_sim = s1
#s1.show_interface()
# Plot many realizations
ax = None
psis_all1 = np.zeros((int(sims_count), len(k_elongs)))
psis_all2 = np.zeros((int(sims_count), len(k_elongs)))

for i in range(sims_count):
    for j, vpol in enumerate(k_elongs):
        # step model
        avg_tr_time = gene_length/vpol
        kelong = (k+l+m+n)/avg_tr_time
        step_sim.set_param("k_elong", kelong)
        step_sim.simulate()
        psis_all1[i,j] = step_sim.get_psi_mean()
        
        # time delays model
        tau1 = avg_tr_time/chain_len*k
        tau2 = avg_tr_time/chain_len*(k+l)
        tau3 = avg_tr_time/chain_len*(k+l+m)
        te1.set_time(tau1)
        te2.set_time(tau2)
        te3.set_time(tau3)
        td_sim.simulate()
        psis_all2[i,j] = td_sim.get_psi_mean()

#psis_diff = psis_all1 - psis_all2
        
labels = ["%.2f" % x for x in k_elongs]        
fig, ax = plt.subplots()
out1 = ax.boxplot(psis_all1, labels = labels, sym = "")
out2 = ax.boxplot(psis_all2,  labels = labels, sym = "")
plt.xticks(rotation=60)

for key, val in out1.items():
    for line in val:
        line.set_color("red")
        line.set_lw(2)

for key, val in out2.items():
    for line in val:
        line.set_color("green")
        line.set_lw(2)

leg_els = []
leg_els.append(ax.axhline(psi_slow, linestyle="--", lw=1.5, color="green", label = "PSI slow"))
leg_els.append(ax.axhline(psi_fast, linestyle="-.", lw=1.5, color="red", label = "PSI fast"))
leg_els.append(ax.axhline(psi_inter, linestyle=":", lw=1.5, color="blue", label = "PSI inter"))
#ax.boxplot(psis_diff, labels = k_elongs)
        
#ax.set_xscale("log")
#ax.axvline(ki, linestyle="--", color="green", label="k_elong=ki")
#ax.axvline(kesc, linestyle="--", color="red", label="k_elong=kesc")
red_l = plt.Line2D([],[], linewidth=3, color="red", label="step model")
green_l = plt.Line2D([],[], linewidth=3, color="green", label = "time delay model")
ax.legend(handles = [red_l, green_l] + leg_els)
ax.set_title("k:%d, l:%d, m:%d, n:%d, mc:%d" % (k, l, m, n, init_mol_count))
ax.set_xlabel("vpol (nt/s)")
ax.set_ylabel("PSI")


#plot analytical solution
psis_analyt = psi_analyticaly(k_elongs, gene_length, k, l, m, n,ki,ks,kesc) 
fig, ax = plt.subplots()
ax.plot(k_elongs, psis_analyt, lw=2)
#ax.set_xscale("log")
kelong = kesc*l
vpol_kesc = gene_length*kelong/(k+l+m+n)
ax.axvline(vpol_kesc, label="kesc=kelong")
ax.axhline(psi_slow, linestyle="--", lw=1.5, color="green", label = "PSI slow")
ax.axhline(psi_fast, linestyle="-.", lw=1.5, color="red", label = "PSI fast")
ax.axhline(psi_inter, linestyle=":", lw=1.5, color="blue", label = "PSI inter")

#plot parameters course
vpol = 50
fig, ax = plt.subplots(figsize=(5,2.5))

avg_tr_time = gene_length/vpol
chain_len = k+l+m+n
tau1 = avg_tr_time/chain_len*k
tau2 = avg_tr_time/chain_len*(k+l)
tau3 = avg_tr_time/chain_len*(k+l+m)

times = [0,tau1,tau2,tau3,avg_tr_time]
kesc_vals = [0,kesc,0,0,0]
ki_vals = [ki,ki,ki,ki,ki]
ks_vals = [0,0,0,ks,ks]

ax.plot(times, ki_vals, color = "green", label ="$ki$",ls="-", lw=2, drawstyle = 'steps-post')
ax.plot(times, ks_vals, color = "red", label ="$ks$",ls="--", lw=2, drawstyle = 'steps-post')
ax.plot(times, kesc_vals, color = "blue", label ="$kesc$",ls=":", lw=2, drawstyle = 'steps-post')
ax.set_title("Parameters")

ax.set_xlabel("time")
ax.set_ylabel("value")
ax.legend()
fig.tight_layout()
