# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:20:04 2019

@author: Timur
"""

import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np

last_n = 5000

s1 = bs.SimParam("CoTranscription_unbranched",
                 100000, 10001,
                 dict(vsyn = 1, k_el=1,  ki = 0.3, ks = 0.3, kesc = 0.3, d=0.04),
                 dict(Incl = 0, Skip = 0))

s1.add_reaction("vsyn", {"p1":1})
for i in range(6):
    ind = i+1
    p1 = "p" + str(ind)
    p2 = "p" + str(ind+1)
    s1.add_reaction("k_el*" + p1, {p1:-1, p2:1} )
    s1.add_reaction("ki *" + p1, {p1:-1, "Incl":1})
s1.add_reaction("ki *" + p2, {p2:-1, "Incl":1})

for i in range(5,8):
    p = "p" + str(i)
    s1.add_reaction("ks*" + p, {p:-1, "Skip":1})

s1.add_reaction("d*Skip", {"Skip":-1})
s1.add_reaction("d*Incl", {"Incl":-1})


s1 = bs.SimParam("CoTranscription_branched_early_inh",
                 10000, 10001,
                 dict(vsyn = 1, k_el=1,  ki = 0.3, ks = 0.3, kesc = 0.3, d=0.04),
                 dict(Incl = 0, Skip = 0))

s1.add_reaction("vsyn", {"p1":1})
s1.add_reaction("kesc*p1",{"e1":1, "p1":-1})
for i in range(6):
    ind = i+1
    p1 = "p" + str(ind)
    p2 = "p" + str(ind+1)
    e1 = "e" + str(ind)
    e2 = "e" + str(ind+1)
    s1.add_reaction("k_el*" + p1, {p1:-1, p2:1} )
    s1.add_reaction("ki *" + p1, {p1:-1, "Incl":1})
    s1.add_reaction("k_el*" + e1, {e1:-1, e2:1})
s1.add_reaction("ki *" + p2, {p2:-1, "Incl":1})

for i in range(5,8):
    p = "p" + str(i)
    e = "e" + str(i)
    s1.add_reaction("ks*" + p, {p:-1, "Skip":1})
    s1.add_reaction("ks*" + e, {e:-1, "Skip":1} )

s1.add_reaction("d*Skip", {"Skip":-1})
s1.add_reaction("d*Incl", {"Incl":-1})



s1 = bs.SimParam("CoTranscription_branched_late_inh",
                 10000, 10001,
                 dict(vsyn = 1, k_el=1,  ki = 0.3, ks = 0.3, kesc = 0.3, d=0.04),
                 dict(Incl = 0, Skip = 0))

s1.add_reaction("vsyn", {"p1":1})
s1.add_reaction("kesc*p3",{"e3":1, "p3":-1})
for i in range(6):
    ind = i+1
    p1 = "p" + str(ind)
    p2 = "p" + str(ind+1)
    e1 = "e" + str(ind)
    e2 = "e" + str(ind+1)
    s1.add_reaction("k_el*" + p1, {p1:-1, p2:1} )
    s1.add_reaction("ki *" + p1, {p1:-1, "Incl":1})
    s1.add_reaction("k_el*" + e1, {e1:-1, e2:1})
s1.add_reaction("ki *" + p2, {p2:-1, "Incl":1})

for i in range(5,8):
    p = "p" + str(i)
    e = "e" + str(i)
    s1.add_reaction("ks*" + p, {p:-1, "Skip":1})
    s1.add_reaction("ks*" + e, {e:-1, "Skip":1} )

s1.add_reaction("d*Skip", {"Skip":-1})
s1.add_reaction("d*Incl", {"Incl":-1})



psi_means =[]
psis = []
incl =[]
skip = []
k_els = np.logspace(-1,2,101)
for k_el in k_els:
    s1.set_param("k_el", k_el)
    s1.simulate()
    psi_means.append(s1.get_psi_mean())
    psis.append(s1.compute_psi(ignore_fraction=0.5)[1])
    incl.append(s1.get_res_col("Incl"))
    skip.append(s1.get_res_col("Skip"))

incl_means = [np.mean(r[-last_n:]) for r in incl]
incl_stds = [np.std(r[-last_n:]) for r in incl]
skip_means = [np.mean(r[-last_n:]) for r in skip]
skip_stds = [np.std(r[-last_n:]) for r in skip]
counts = np.add(incl_means, skip_means)
psi_stds = [np.std(r[-last_n:]) for r in psis]
psi_stds_stabl = [np.std(np.arcsin(np.sqrt(r[-last_n:]))) for r in psis]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.semilogx(k_els, psi_means)

psi_std_basic_model = sth.tmp_simulate_std_gillespie(counts, psi_means, var_stab=False)
psi_std_basic_model_stabl = sth.tmp_simulate_std_gillespie(counts, psi_means, var_stab=True)


fig = plt.figure()
ax = fig.add_subplot(1,3,1)
ax.scatter(psi_stds, psi_std_basic_model)
ax.set_xlabel("CoTranscr. Splicing: std(PSI)")
ax.set_ylabel("Basic Spl. model: std(PSI)")
ax.set_title("Without stabilizing")
std_max = max(psi_std_basic_model)
std_max *= 1.1
ax.plot([0,std_max], [0, std_max], c = "red")

ax = fig.add_subplot(1,3,2)
ax.scatter(psi_stds_stabl, psi_std_basic_model_stabl)
ax.set_xlabel("CoTranscr. Splicing: std(PSI)")
ax.set_ylabel("Basic Spl. model: std(PSI)")
ax.set_title("With stabilizing")
std_max = max(psi_std_basic_model)
std_max *= 1.1
ax.plot([0,std_max], [0, std_max], c = "red")