# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:54:52 2020

@author: Timur
"""


import bioch_sim as bs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
import scipy as sp
import numpy as np
import sympy as sy
from sympy.parsing.sympy_parser import parse_expr



sim_count = 100
init_mol_count = 100


fig = "5A"

if fig == "5A":
    s = bs.get_exmpl_CoTrSpl()
    init_sp = "P000"
    s.set_color("P100", "magenta")
    s.set_color("P000_inh", "orange")
    s.set_param("rbp_pos", 330)
    prod_to_show = ("Incl", "Skip",  "P000", "P100", "P000_inh", "P100_inh", "ret" )
    runtime = 25

if fig == "5B":
    s = bs.get_exmpl_CoTrSpl_simple()["td_m"]
    init_sp = "mRNAinh"
    s.set_color("mRNAinh", "orange")
    prod_to_show = []
    runtime = 7

s.set_color("Incl", "green")
s.set_color("Skip", "red")

s.set_runtime(runtime)
s.set_init_species(init_sp, init_mol_count)
s.simulate(sim_count, verbose = False)

#s.plot_series(products=("Incl", "Skip",  "P000", "P001", "P110_inh", "P011_inh", "P010_inh", "P111", "P111_inh", "ret" ))
ax = s.plot_series(products=prod_to_show, scale=0.7)
s.annotate_timeEvents(ax, 12)
#ax.tight_layout()
#s.plot_series()
#s.plot_course(products=["Incl", "Skip",  "P000", "P001" ])
#s.plot_parameters()
s.set_runtime(1e4)

s.simulate(sim_count, verbose = False)
res = s.get_res_by_expr_2("Incl/(Incl+Skip)", t_bounds = (s.get_runtime()*0.999,np.inf), series=True )
res = np.array(res)
fig, ax = plt.subplots()
#ax.plot(res.T, c="blue", lw=0.2)
ax.hist(res[:,-1])
ax.set_title("PSI distribution")
ax.set_xlabel("PSI")


s.set_runtime(2000)

def compare_to_binomial(model, par="vpol", vals=np.linspace(10,1000,50), init_sp = "mRNA",
                        init_mol_counts = [20,200,500], s_count = 100):
    s = model
    
    psi_means = np.zeros(len(vals) * len(init_mol_counts))
    psi_stds = np.array(psi_means)
    counts = np.array(psi_means)
    i=0
    for m_count in init_mol_counts:
        s.set_init_species(init_sp, m_count)
        for  v in vals:
            s.set_param(par, v)
            s.simulate(s_count)
            incls = s.get_res_col("Incl", series=True)[:,-1]
            skips = s.get_res_col("Skip", series=True)[:,-1]
            counts[i] = np.mean(skips + incls)
            
            psis_end = incls/(incls+skips)
            psi_means[i] = np.mean(psis_end)
            psi_stds[i] = np.std(psis_end)
            i += 1
        
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    binom = sp.stats.binom.std(counts, psi_means)/counts
    cmap = plt.cm.get_cmap('RdYlBu')
    paths = ax.scatter(psi_stds, binom, s = (psi_means+0.5)*100, c = counts, cmap = cmap, alpha=0.35)
    cbar = fig.colorbar(paths, ax = ax)
#    cbar = ax.figure.colorbar(None, ax=ax)
    cbar.ax.set_ylabel("counts", rotation=-90, va="bottom")
    ax.set_ylabel("std(PSI), Binomial")
    ax.set_xlabel("std(PSI), Gillespie")
    ax.set_title(str(s_count) + " simulations per point")
    ax.plot([0,np.max(binom)],[0,np.max(binom)], c="r")
    return dict(counts = counts, psi_means= psi_means, psi_stds = psi_stds,ax= ax, fig=fig)

res = compare_to_binomial(s, "rbp_pos", np.linspace(150,300,30), "P000", [10,50,100], s_count= 2000)

