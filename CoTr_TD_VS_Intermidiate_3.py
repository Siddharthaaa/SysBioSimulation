# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:13:21 2019

@author: Timur
"""



import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np

sims_count = 5000

gene_length = 800 #nt
init_mol_count = 50

analytical = False

fontsize=10
plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize)

plt.rc("font", size=fontsize)

#select model:
# 1: unbranched few steps
# 2: unbranched many steps

# 3: branched late inh few steps
# 4: branched late inh many steps

# 5: branched early inh few steps
# 6: branched early inh many steps

# 7: branched early inh bell few steps
# 8: branched early inh bell many steps

model_id = "test"

vpols = np.logspace(0,3,100)

kesc = 1
vpol = 50

kesc_r = 0

class ModelTopology(object):
    def __init__(self):
        pass
    
def get_models(model_id="test", vpol = 50, runtime=1e4):
    
    kesc_r = 0
    if model_id == "test":
        l = 8
        m1 = 0
        m2 = 8
        n = 3
        k = 2
        
        ki = 0.5
        ks = 1
        kesc = 0.8
        kesc_r = 0
    
    if model_id == "test2":
        l = 80
        m1 = 20
        m2 = 40
        n = 30
        k = 30
        ki = 0.1
        ks = 1
        kesc = 0.2
        
    
    if model_id == 1:
        l = 8
        m1 = 0
        m2 = 0
        n = 2
        k = 0
        ki = 5e-2
        ks = 5e-1
        kesc = 0
        
    if model_id == 2:
        l = 80
        m1 = 0
        m2 = 0
        n = 20
        k = 0
        ki = 5e-2
        ks = 5e-1
        kesc = 0
        
    if model_id == 3:
        
        l=8
        k=0
        m1 = 2
        m2 = 3
        n = 2
        ki = 5e-2
    #    ki = 1e-1
    #    ki = 5e-1
        ks = 5e-3
        kesc = 0.5
        kesc_r = 0
    
    if model_id == 4:
        l=80
        k=0
        m1 = 20
        m2 = 30
        n = 20
        ki = 5e-2
        ks = 1e-3
    #    kesc = 0.5
        
    
    if model_id == 5:
        
        l=8
        k=0
        m1 = 0
        m2 = 1
        n=3
        
        ki = 1e-1
        ks = 1e-2
        kesc = 0.5
        k_elongs = np.logspace(0,3,40)
        
    #    psi_slow = ki/(ki+kesc)
    #    psi_fast = ki/(ki+ks)
    #    psi_inter = ki/(ki+kesc)
    
    if model_id == 6:
        
        l=80
        k=0
        m1 = 0
        m2 = 10
        n=30
        
        ki = 1e-1
        ks = 1e-2
        kesc = 0.5
        k_elongs = np.logspace(0,3,40)
        
    
    if model_id == 7:
        l = 8
        k=0
        m1 = 0
        m2 = 1
        n = 2
        
        ki = 1e-1
        ks = 2e-1
        kesc = 2e-1
        k_elongs = np.logspace(0,3.2,40)
    
    if model_id == 8:
        l = 80
        k=0
        m1 = 0
        m2 = 10
        n = 20
        
        ki = 1e-1
        ks = 2e-1
        kesc = 2e-1
        k_elongs = np.logspace(0,3,40)
        
        
    avg_tr_time = gene_length/vpol
    kelong = l/avg_tr_time
    s1 = bs.SimParam("CoTrSpl_general",
                     runtime, 10001,
                     dict(k_elong=kelong,  ki = ki, ks = ks, kesc = kesc, kesc_r = kesc_r),
                     dict(p1=init_mol_count, Incl = 0, Skip = 0))
    
    # upper chain
    for i in range(1, l):
        p1 = "p" + str(i)
        p2 = "p" + str(i+1)
        s1.add_reaction("k_elong*" + p1, {p1:-1, p2:1} )
        if i > k:
            s1.add_reaction("ki *" + p1, {p1:-1, "Incl":1})
    if k < l:
        s1.add_reaction("ki *" + p2, {p2:-1, "Incl":1})
    
    # lower chain
    for i in range(m1+1, l):
        e1 = "e" + str(i)
        e2 = "e" + str(i+1)
        s1.add_reaction("k_elong*" + e1, {e1:-1, e2:1})
    
    #escape transitions
    for i in range(m1, m1+m2):
        p = "p" + str(i+1)
        e = "e" + str(i+1)
        s1.add_reaction("kesc*" + p, {p:-1, e:1})
        if (kesc_r > 0):
            s1.add_reaction("kesc_r*" + e, {p:1, e:-1})
    
    #last skipping steps
    for i in range(l-n, l):
        p = "p" + str(i+1)
        e = "e" + str(i+1)
        s1.add_reaction("ks*" + p, {p:-1, "Skip":1})
        s1.add_reaction("ks*" + e, {e:-1, "Skip":1})
        
    
    s1.compile_system()
    #s1.draw_pn(engine="dot", rates=False)
    step_sim = s1
    
    
    s1 = bs.SimParam("CoTrSpl_general_TD",
                     runtime, 10001,
                     dict(ki=0, ki_on = ki,
                          ks = 0, ks_on = ks,
                          kesc = 0, kesc_on=kesc, kesc_r = kesc_r),
                     dict(mRNA = init_mol_count, Incl = 0, Skip = 0))
    
    s1.add_reaction("mRNA*ki", {"Incl":1, "mRNA":-1})
    s1.add_reaction("mRNA*ks", {"Skip":1, "mRNA":-1})
    s1.add_reaction("mRNA*kesc", {"mRNAbr":1, "mRNA":-1})
    if(kesc_r > 0):
        s1.add_reaction("mRNAbr*kesc_r", {"mRNAbr":-1, "mRNA":1})
        
    s1.add_reaction("mRNAbr*ks", {"Skip":1, "mRNAbr":-1})
    
    t_per_step = avg_tr_time /l
    
    tau1 = t_per_step * k
    tau2 = t_per_step * m1
    tau3 = t_per_step * (m1 + m2)
    tau4 = t_per_step * (l-n)
    te1 = bs.TimeEvent(tau1, "ki = ki_on")
    te2 = bs.TimeEvent(tau2, "kesc=kesc_on")
    te3 = bs.TimeEvent(tau3, "kesc=0; kesc_r = 0")
    te4 = bs.TimeEvent(tau4, "ks=ks_on")
    s1.add_timeEvent(te1)
    s1.add_timeEvent(te2)
    s1.add_timeEvent(te3)
    s1.add_timeEvent(te4)
    
    td_sim = s1

    return step_sim, td_sim

#TODO
def psi_analyticaly(vpol, gene_length, l, m1, m2, k, n,ki,ks,kesc, kesc_r):
    avg_tr_time = gene_length/vpol
    kelong = l/avg_tr_time
    t_per_step = 1/kelong
    taus =[]
    taus.append(t_per_step * k)
    taus.append(t_per_step * m1)
    taus.append(t_per_step * (m1 + m2))
    taus.append(t_per_step * (l-n))
    taus.append(np.inf)
    indx = np.argsort(taus)
    p=1
    pi = 0
    ki_b = ks_b = kesc_b = 0
    t = 0
    for i in indx:
        tau = taus[i]
        td = tau-t
        t = tau
        A = td*(ki_b + ks_b + kesc_b)
        pt = p*(1-np.exp(-A))
        p -= pt
        ksum = (ki_b + ks_b + kesc_b)
        pi += pt*ki_b/ksum if ksum >0 else 0   
        if(i == 0):
            ki_b = ki
        elif(i == 1):
            #TODO desc_r does not function proper
            kesc_b = kesc*(kesc/(kesc+kesc_r)) if kesc >0 else 0
        elif(i ==2):
            kesc_b = 0
        elif(i==3):
            ks_b = ks
    
    
    return pi
#
#psi_inter =ki/(ki+ kesc)
##psi_inter =ki/(ki+kesc)
# 
#if(k >= m1):
#    psi_slow = ki/(ki+kesc)
#else:
#    psi_slow = ki/ki
#
#psi_fast = ki/(ki+ks)

vpol = 50. # nt/s
#kesc = 10
runtime = 10000

#s1.draw_pn(engine="dot", rates=False)

#s1.show_interface()
# Plot many realizations

ax = None

#fig, ax = plt.subplots()


models_res = {}
for model in [1,3,5,7,"test"]:
    psis_all1 = np.zeros((int(sims_count), len(vpols)))
    psis_all2 = np.zeros((int(sims_count), len(vpols)))    
    
    for j, vpol in enumerate(vpols):
        step_sim, td_sim = get_models(model, vpol = vpol)
    
        for i in range(sims_count):
            # step model
            avg_tr_time = gene_length/vpol
#            kelong = l/avg_tr_time
#            step_sim.set_param("k_elong", kelong)
            step_sim.simulate()
            incl = step_sim.get_res_col("Incl")[-1]
            skip = step_sim.get_res_col("Skip")[-1]
            
            psis_all1[i,j] = incl/(incl+skip)
            
#            t_per_step = avg_tr_time / l
#            # time delays model
#            tau1 = t_per_step * k
#            tau2 = t_per_step * m1
#            tau3 = t_per_step * (m1 + m2)
#            tau4 = t_per_step * (l-n)
#            
#            te1.set_time(tau1)
#            te2.set_time(tau2)
#            te3.set_time(tau3)
#            te4.set_time(tau4)
            td_sim.simulate()
            incl = td_sim.get_res_col("Incl")[-1]
            skip = td_sim.get_res_col("Skip")[-1]
            psis_all2[i,j] = incl/(incl+skip)
    
    psis_stds = np.std(psis_all1, axis=0)
    psis_means = np.mean(psis_all1, axis=0)
    models_res[model] = [psis_means, psis_stds]
    
#    ax.scatter(psis_means, psis_stds, marker="o", alpha=0.5, label="model: %s" % str(model))

fig, ax = plt.subplots()
for m, v in models_res.items():
    ax.scatter(v[0], v[1], alpha=0.6, label="model: %s" % str(m))
ax.legend()
ax.set_ylabel("std(PSI)")
ax.set_xlabel("mean(PSI)")
ax.set_title("Noise (mc:%d, sim_count: %d)" % (init_mol_count, sims_count ))
#plot noises

