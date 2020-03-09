# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:13:21 2019

@author: Timur
"""



import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth
from scipy.stats import binom

import numpy as np

sims_count = 100

gene_length = 800 #nt
trscrpt_start = 700 #nt
trscrpt_end = 1000 #nt
tr_len = trscrpt_end - trscrpt_start

init_mol_count = 50

variance_analysis = True

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

model_id = 1

vpols = np.logspace(0,3,40)

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
        
        
    
    
    models = bs.coTrSplCommitment(vpol=vpol, tr_len=tr_len, l=l, m1=m1, m2=m2, k=k, n=n,
                    ki=ki, ks=ks, kesc=kesc, kesc_r=0)
    td_sim = models["td_m"]
    step_sim = models["step_m"]
    td_sim = models["td_m"]
    psi_f = models["psi_analytic_f"]
    return step_sim, td_sim, psi_f

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

if (variance_analysis):
    fig, ax = plt.subplots()
    for mol_count, ls  in zip([10, 30, 100,1000],  ["-","--","-.",":"]):
        
        models_res = {}
        for model in [1,3,5,7,"test"]:
            psis_all1 = np.zeros((int(sims_count), len(vpols)))
            psis_all2 = np.zeros((int(sims_count), len(vpols)))    
            step_sim, td_sim = get_models(model, vpol = vpol)
            td_sim.set_init_species("mRNA", mol_count)
            for j, vpol in enumerate(vpols):
                td_sim.set_param("vpol", vpol)
                for i in range(sims_count):
   
                    td_sim.simulate()
                    incl = td_sim.get_res_col("Incl")[-1]
                    skip = td_sim.get_res_col("Skip")[-1]
                    
                    psis_all1[i,j] = incl/(incl+skip)
                    
   
            psis_stds = np.std(psis_all1, axis=0)
            psis_means = np.mean(psis_all1, axis=0)
            models_res[model] = [psis_means, psis_stds]
        
    #    ax.scatter(psis_means, psis_stds, marker="o", alpha=0.5, label="model: %s" % str(model))
    
        leg_els =[]
        for (m, v), c in zip(models_res.items(),["red", "blue", "green", "orange","purple"]):
            leg_els.append(ax.scatter(v[0], v[1], alpha=0.6, color = c
#                       ,label="model: %s" % str(m)
                       ))
        psis_th = np.linspace(0,1,100)
        b_stds = [binom.std(mol_count, p)/mol_count for p in psis_th]
        ax.plot(psis_th, b_stds, color = "black", lw=2,ls=ls,
                alpha = 0.7,
                label = "binom. (mc:%d)" % mol_count )
    leg1 = ax.legend(loc=2)
    black_dot, = plt.plot([], "o", color = "black")
#    plt.legend([red_dot],["AAAA"])
    leg2 = plt.legend([black_dot], ["models"], loc=1)
    ax.add_artist(leg1)
    ax.set_ylabel("std(PSI)")
    ax.set_xlabel("mean(PSI)")
    ax.set_title("Noise ( sim_count: %d)" % sims_count)
    #plot noises

