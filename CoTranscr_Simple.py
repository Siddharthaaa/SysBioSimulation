# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:20:04 2019

@author: Timur
"""

import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np

n = 4
m = 3

kesc = 0.3
kesc = 10
runtime = 5e4
last_n = 5000


s1 = bs.SimParam("CoTranscription_unbranched",
                 runtime, 10001,
                 dict(vsyn = 1, k_el=1,  ki = 0.3, ks = 0.3, kesc = kesc, d=0.04),
                 dict(Incl = 0, Skip = 0))

s1.add_reaction("vsyn", {"p1":1})
for i in range(1,n+m):
    p1 = "p" + str(i)
    p2 = "p" + str(i+1)
    s1.add_reaction("k_el*" + p1, {p1:-1, p2:1} )
    s1.add_reaction("ki *" + p1, {p1:-1, "Incl":1})
s1.add_reaction("ki *" + p2, {p2:-1, "Incl":1})

for i in range(n,n+m):
    p = "p" + str(i+1)
    s1.add_reaction("ks*" + p, {p:-1, "Skip":1})

s1.add_reaction("d*Skip", {"Skip":-1})
s1.add_reaction("d*Incl", {"Incl":-1})
s_unbr = s1



s1 = bs.SimParam("CoTranscription_branched_early_inh",
                 runtime, 10001,
                 dict(vsyn = 1, k_el=1,  ki = 0.3, ks = 0.3, kesc = kesc, d=0.04),
                 dict(Incl = 0, Skip = 0))

s1.add_reaction("vsyn", {"p1":1})
s1.add_reaction("kesc*p1",{"e1":1, "p1":-1})
for i in range(1,n+m):
    p1 = "p" + str(i)
    p2 = "p" + str(i+1)
    e1 = "e" + str(i)
    e2 = "e" + str(i+1)
    s1.add_reaction("k_el*" + p1, {p1:-1, p2:1} )
    s1.add_reaction("ki *" + p1, {p1:-1, "Incl":1})
    s1.add_reaction("k_el*" + e1, {e1:-1, e2:1})
s1.add_reaction("ki *" + p2, {p2:-1, "Incl":1})

for i in range(n,n+m):
    p = "p" + str(i+1)
    e = "e" + str(i+1)
    s1.add_reaction("ks*" + p, {p:-1, "Skip":1})
    s1.add_reaction("ks*" + e, {e:-1, "Skip":1} )

s1.add_reaction("d*Skip", {"Skip":-1})
s1.add_reaction("d*Incl", {"Incl":-1})
s_br_early = s1


s1 = bs.SimParam("CoTranscription_branched_late_inh",
                 runtime, 10001,
                 dict(vsyn = 1, k_el=1,  ki = 0.3, ks = 0.3, kesc = kesc, d=0.04),
                 dict(Incl = 0, Skip = 0))

s1.add_reaction("vsyn", {"p1":1})
s1.add_reaction("kesc*p3",{"e3":1, "p3":-1})
for i in range(1, n+m):
    p1 = "p" + str(i)
    p2 = "p" + str(i+1)
    e1 = "e" + str(i)
    e2 = "e" + str(i+1)
    s1.add_reaction("k_el*" + p1, {p1:-1, p2:1} )
    s1.add_reaction("ki *" + p1, {p1:-1, "Incl":1})
    s1.add_reaction("k_el*" + e1, {e1:-1, e2:1})
s1.add_reaction("ki *" + p2, {p2:-1, "Incl":1})

for i in range(n,n+m):
    p = "p" + str(i+1)
    e = "e" + str(i+1)
    s1.add_reaction("ks*" + p, {p:-1, "Skip":1})
    s1.add_reaction("ks*" + e, {e:-1, "Skip":1} )

s1.add_reaction("d*Skip", {"Skip":-1})
s1.add_reaction("d*Incl", {"Incl":-1})
s_br_late = s1

s_unbr.draw_pn(engine="dot", rates = True)


k_els = np.logspace(-1,2,201)
ax = s_unbr.plot_par_var_1d(par = "k_el", vals = k_els, label = "Unbranched", ax = None,
                            func=s_unbr.get_psi_mean, ignore_fraction = 0.5)
ax = s_br_early.plot_par_var_1d(par = "k_el", vals = k_els, label = "Branch, early inh.",
                                ax = ax, func=s_br_early.get_psi_mean, ignore_fraction = 0.5)
ax = s_br_late.plot_par_var_1d(par = "k_el", vals = k_els, label = "Branch, late inh.",
                               ax = ax, func=s_br_late.get_psi_mean, ignore_fraction = 0.5)
ax.set_xscale("log")
#ax.set_label("Weak inhibition")
#ax.set_title("Weak inhibition")
ax.set_title("Strong inhibition")
ax.legend()


#
#psi_means =[]
#psis = []
#incl =[]
#skip = []
#
#for k_el in k_els:
#    s1.set_param("k_el", k_el)
#    s1.simulate()
#    psi_means.append(s1.get_psi_mean())
#    psis.append(s1.compute_psi(ignore_fraction=0.5)[1])
#    incl.append(s1.get_res_col("Incl"))
#    skip.append(s1.get_res_col("Skip"))
#
#incl_means = [np.mean(r[-last_n:]) for r in incl]
#incl_stds = [np.std(r[-last_n:]) for r in incl]
#skip_means = [np.mean(r[-last_n:]) for r in skip]
#skip_stds = [np.std(r[-last_n:]) for r in skip]
#counts = np.add(incl_means, skip_means)
#psi_stds = [np.std(r[-last_n:]) for r in psis]
#psi_stds_stabl = [np.std(np.arcsin(np.sqrt(r[-last_n:]))) for r in psis]
#
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.semilogx(k_els, psi_means)
#
#psi_std_basic_model = sth.tmp_simulate_std_gillespie(counts, psi_means, var_stab=False)
#psi_std_basic_model_stabl = sth.tmp_simulate_std_gillespie(counts, psi_means, var_stab=True)
#
#
#fig = plt.figure()
#ax = fig.add_subplot(1,3,1)
#ax.scatter(psi_stds, psi_std_basic_model)
#ax.set_xlabel("CoTranscr. Splicing: std(PSI)")
#ax.set_ylabel("Basic Spl. model: std(PSI)")
#ax.set_title("Without stabilizing")
#std_max = max(psi_std_basic_model)
#std_max *= 1.1
#ax.plot([0,std_max], [0, std_max], c = "red")
#
#ax = fig.add_subplot(1,3,2)
#ax.scatter(psi_stds_stabl, psi_std_basic_model_stabl)
#ax.set_xlabel("CoTranscr. Splicing: std(PSI)")
#ax.set_ylabel("Basic Spl. model: std(PSI)")
#ax.set_title("With stabilizing")
#std_max = max(psi_std_basic_model)
#std_max *= 1.1
#ax.plot([0,std_max], [0, std_max], c = "red")