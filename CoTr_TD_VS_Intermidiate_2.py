# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:13:21 2019

@author: Timur
"""



import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np


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

sims_count = 10

#gene_length = 800 #nt
trscrpt_start = 700 #nt site of U1 of exon2
trscrpt_end = 1000 #nt site of U1 of exon3
tr_len = trscrpt_end - trscrpt_start

init_mol_count = 100


deterministic_solution = True
stochastic_analysis = False
analytical = True
plot_derivations = False

fontsize=10
plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize)

plt.rc("font", size=fontsize)


vpols = np.logspace(0,3.2,50)

kesc = 1
vpol = 50

kesc_r = 0

vpol_oi = {} # vpols of interest. key: vpol val, value: name/short descr.

class ModelTopology(object):
    def __init__(self):
        pass
    
#def get_models(model_id="test", vpol = 50, runtime=1e4):
    
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
    len_per_step = tr_len/l
    vpolOI = len_per_step*(l-n) * ki
    vpol_oi[vpolOI] = "vpol: $ki*\\tau=1$"

    
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
    
    len_per_step = tr_len/l
    vpolOI = len_per_step*(m1) * ki
    vpol_oi[vpolOI] = "vpol: $ki*\\tau1=1$"
    vpolOI = len_per_step*(m2) * (kesc + ki)
    vpol_oi[vpolOI] = "vpol: $(kesc + ki)*\\tau2=1$"

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


psi_inter =ki/(ki+ kesc)
#psi_inter =ki/(ki+kesc)
 
if(k >= m1):
    psi_slow = ki/(ki+kesc)
else:
    psi_slow = ki/ki

psi_fast = ki/(ki+ks)

vpol = 50. # nt/s
#kesc = 10
runtime = 10000


s1 = bs.SimParam("CoTrSpl_general",
                 runtime, 10001,
                 dict(vpol = vpol,
                      l = l,
                      tr_len = tr_len,
                      k_elong="vpol*l/tr_len",
                      ki = ki, ks = ks,
                      kesc = kesc, kesc_r = kesc_r),
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
if (m1>m2):
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
                 dict(vpol = vpol,
                         l = l, tr_len = tr_len,
                         ki=0, ki_on = ki,
                      ks = 0, ks_on = ks,
                      kesc = 0, kesc_on=kesc, kesc_r = kesc_r),
                 dict(mRNA = init_mol_count, Incl = 0, Skip = 0))

s1.add_reaction("mRNA*ki", {"Incl":1, "mRNA":-1})
s1.add_reaction("mRNA*ks", {"Skip":1, "mRNA":-1})
s1.add_reaction("mRNA*kesc", {"mRNAbr":1, "mRNA":-1})
if(kesc_r > 0):
    s1.add_reaction("mRNAbr*kesc_r", {"mRNAbr":-1, "mRNA":1})
    
s1.add_reaction("mRNAbr*ks", {"Skip":1, "mRNAbr":-1})

tau1 = "tr_len/l/vpol * %d" % k
tau2 = "tr_len/l/vpol * %d" % m1
tau3 = "tr_len/l/vpol * %d" % (m1 + m2)
tau4 = "tr_len/l/vpol * %d" % (l-n)
te1 = bs.TimeEvent(tau1, "ki = ki_on", "Incl. on")
te2 = bs.TimeEvent(tau2, "kesc=kesc_on", "Esc. on")
te3 = bs.TimeEvent(tau3, "kesc=0; kesc_r = 0", "Esc. off")
te4 = bs.TimeEvent(tau4, "ks=ks_on", "Skip. on")
s1.add_timeEvent(te1)
s1.add_timeEvent(te2)
s1.add_timeEvent(te3)
s1.add_timeEvent(te4)

#s1.draw_pn(engine="dot", rates=False)
td_sim = s1
#return  step_sim, td_sim

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



# Plot many realizations
ax = None
psis_all1 = np.zeros((int(sims_count), len(vpols)))
psis_all2 = np.zeros((int(sims_count), len(vpols)))

psis_det_step = np.zeros( len(vpols))
psis_det_td = np.zeros( len(vpols))
#td_sim.show_interface()
#step_sim.show_interface()

if deterministic_solution:
    fig, ax = plt.subplots()
    vpols_analyt = vpols[0::4]
    psis_analyt = [psi_analyticaly(vpol2, tr_len, l, m1, m2, k, n,ki,ks,kesc,kesc_r) for vpol2 in vpols_analyt]
    for j, vpol in enumerate(vpols):
       
        step_sim.set_param("vpol", vpol)
        td_sim.set_param("vpol", vpol)
        
        step_sim.simulate(ODE=True)
        td_sim.simulate(ODE=True)
        
        incl = step_sim.get_res_col("Incl", method="ODE")[-1]
        skip = step_sim.get_res_col("Skip", method="ODE")[-1]
        psis_det_step[j] = incl/(incl+skip)
        
        incl = td_sim.get_res_col("Incl", method="ODE")[-1]
        skip = td_sim.get_res_col("Skip", method="ODE")[-1]
        psis_det_td[j] = incl/(incl+skip)
        
    #    ax.plot(vpols, psis_all2.T, lw=0.1, alpha=1, color="grey", antialiased=True)
#    ax.plot(vpols, psis_det_step, lw=3, color="red", label= "step model" )
    ax.plot(vpols, psis_det_td, lw=3, color="green", label="time delay model")
    ax.plot(vpols_analyt, psis_analyt, "bo",ms=8,  label="analytic solution")
    
    ax.set_title("Model: %s" % str(model_id))
    ax.set_xscale("log")
    ax.set_xlabel("vpol (nt/s)")
    ax.set_ylabel("PSI")
        
    ax.axhline(psi_slow, linestyle="--", lw=1.5, color="green", label = "PSI slow")
    ax.axhline(psi_fast, linestyle="-.", lw=1.5, color="red", label = "PSI fast")
#    ax.axhline(psi_inter, linestyle=":", lw=1.5, color="blue", label = "PSI medium")
    
    for key,val in vpol_oi.items():
        ax.axvline(key, ls = ":", lw=2, label = val)
    ax.legend()
    
    
if stochastic_analysis:
    for j, vpol in enumerate(vpols):
       
        step_sim.set_param("vpol", vpol)
        td_sim.set_param("vpol", vpol)
        
        for i in range(sims_count):
            # step model
            
            step_sim.simulate()
            incl = step_sim.get_res_col("Incl")[-1]
            skip = step_sim.get_res_col("Skip")[-1]
            
            psis_all1[i,j] = incl/(incl+skip)
            
           
            td_sim.simulate()
            incl = td_sim.get_res_col("Incl")[-1]
            skip = td_sim.get_res_col("Skip")[-1]
            psis_all2[i,j] = incl/(incl+skip)
    
    #compare noise
    psis_stds = np.std(psis_all2, axis=0)
    counts = np.ones(psis_all2.shape[-1]) * init_mol_count
    psi_stds_exp = sth.tmp_simulate_std_gillespie(counts, np.mean(psis_all2, axis=0))
    fig, ax = plt.subplots()
    ax.scatter(psis_stds, psi_stds_exp)
    ax.plot([0,max(psis_stds)], [0,max(psis_stds)])
    ax.set_title("Noise comparison")
    ax.set_ylabel("Simple model")
    ax.set_xlabel("Tested model")
    
    #psis_diff = psis_all1 - psis_all2
            
    labels = ["%.2f" % x for x in vpols]        
    fig, axs = plt.subplots(2,1, figsize=(3,7))
    ax = axs[0]
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
    leg_els.append(ax.axhline(psi_inter, linestyle=":", lw=1.5, color="blue", label = "PSI medium"))
    #ax.boxplot(psis_diff, labels = vpols)
            
    #ax.set_xscale("log")
    #ax.axvline(ki, linestyle="--", color="green", label="k_elong=ki")
    #ax.axvline(kesc, linestyle="--", color="red", label="k_elong=kesc")
    red_l = plt.Line2D([],[], linewidth=3, color="red", label="step model")
    green_l = plt.Line2D([],[], linewidth=3, color="green", label = "time delay model")
    ax.legend(handles = [red_l, green_l] + leg_els, fontsize=fontsize)
    ax.set_title("l:%d, m1:%d, m2:%d k:%d, n:%d, mc:%d" % (l, m1, m2, k, n, init_mol_count))
    ax.set_xlabel("vpol (nt/s)")
    ax.set_ylabel("PSI")
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.setp(ax.get_xticklabels(), visible=False)
    #every_nth =8
    #for num, label in enumerate(ax.xaxis.get_ticklabels()):
    #    if num % every_nth != 0:
    #        label.set_visible(False)
    #        label.set_rotation(140)
    #plt.setp(ax.get_yticklabels(), visible=False)


if (analytical):
#plot analytical solution
    psis_analyt = [psi_analyticaly(vpol2, tr_len, l, m1, m2, k, n,ki,ks,kesc,kesc_r) for vpol2 in vpols]
    vpols_log = np.log10(vpols)
#    vpols_log = vpols
    derv_1_log = sth.numerical_derivation(vpols_log, psis_analyt)
    derv_2_log = sth.numerical_derivation(vpols_log, derv_1_log)
    zeros_log = sth.get_zero_points(vpols_log, derv_2_log)
    zeros_log_tr = 10**zeros_log
    derv_1 = sth.numerical_derivation(vpols, psis_analyt)
    derv_2 = sth.numerical_derivation(vpols, derv_1)
    zeros = sth.get_zero_points(vpols, derv_2)
#   
#    fig, ax = plt.subplots()
    fig, ax = plt.subplots()
    ax.plot(vpols, psis_analyt, lw=3)
    ax_twin = ax.twinx()
    if plot_derivations:
        ax_twin.plot(vpols, derv_1, lw = 0.5, label = "derivation 1")
        ax_twin.plot(vpols, derv_2, lw = 0.5, label = "derivation 2")
    ax.set_xscale("log")
    ax_twin.set_xscale("log")
    ax_twin.axhline(0, ls="--", lw=0.7, color="black")
    ax_twin.legend(fontsize=fontsize)
    kelong = kesc*l
    for z in zeros:
        ax_twin.axvline(z, ls="-", lw=0.7, color="green" )
#        ax_twin.annotate("%.2f" % z, (z,0), xytext=(z-0.1,-0.1),
#                         arrowprops={"arrowstyle": "->"}, color="green")
    for z in zeros_log_tr:
        ax_twin.axvline(z, ls="-", lw=0.7, color="red")
        
    green_l = plt.Line2D([],[], linewidth=0.7, linestyle="-", color="green",
                         label="infl.p. lin(%s)" % (", ".join("%.0f" % z for z in zeros)))
    red_l = plt.Line2D([],[], linewidth=0.7, linestyle="-", color="red",
                       label = "infl.p. log(%s)" % (", ".join("%.0f" % z for z in zeros_log_tr)))
        
    ax.axhline(psi_slow, linestyle="--", lw=1.5, color="green", label = "PSI slow")
    ax.axhline(psi_fast, linestyle="-.", lw=1.5, color="red", label = "PSI fast")
    ax.axhline(psi_inter, linestyle=":", lw=1.5, color="blue", label = "PSI medium")
    ax.legend(fontsize=fontsize)
    ax.set_title("analytic")
#    ax_twin.legend([red_l, green_l])
    ax_twin.legend(handles = [red_l, green_l], fontsize=fontsize)
    
    
#fig.legend(loc=7)
    
#plot parameters course
vpol = 50
fig, ax = plt.subplots(figsize=(5,2.5))

avg_tr_time = tr_len/vpol
t_per_step = avg_tr_time /l
tau1 = t_per_step * k
tau2 = t_per_step * m1
tau3 = t_per_step * (m1 + m2)
tau4 = t_per_step * (l-n)

att = avg_tr_time
ax.plot([0,tau1,att], [0,ki,ki], color = "green", label ="$ki$",ls="-", lw=2, drawstyle = 'steps-post')
ax.plot([0,tau4,att], [0,ks,ks], color = "red", label ="$ks$",ls="--", lw=2, drawstyle = 'steps-post')
ax.plot([0,tau2,tau3,att], [0,kesc,0,0], color = "blue", label ="$kesc$",ls=":", lw=2, drawstyle = 'steps-post')
ax.set_title("Parameters (vpol=%.1f)" % vpol)
ax.set_xlabel("time")
ax.set_ylabel("value")
ax.legend(fontsize=fontsize)
fig.tight_layout()
