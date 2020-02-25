# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:54:52 2020

@author: Timur
"""


import bioch_sim as bs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np
import sympy as sy
from sympy.parsing.sympy_parser import parse_expr


sim_count = 500

s = bs.get_exmpl_CoTrSpl()
s.set_color("Incl", "green")
s.set_color("Skip", "red")

s.set_runtime(80)
s.set_init_species("P000", 50)
s.simulate(sim_count)

s.plot_series(products=("Incl", "Skip",  "P000", "P001" ))
#s.plot_course(products=["Incl", "Skip",  "P000", "P001" ])

res = s.get_res_by_expr_2("Incl/(Incl+Skip)", t_bounds = (79,np.inf), series=True )
res = np.array(res)
fig, ax = plt.subplots()
ax.plot(res.T, c="black", lw=0.2)


psi_std = np.std(res[:,-1])

def tmp_compare_binomial_gillespie(counts, psi_means, exact_counts=False):
    fig = plt.figure()
    binom = sp.stats.binom.std(counts, psi_means)/counts
    gillespie = tmp_simulate_std_gillespie(counts, psi_means, runtime=1e5, exact_counts=exact_counts)
#    gillespie = tmp_simulate_std_binomial(counts, psi_means)
#    gillespie = simulate_dropout(counts, psi_means,0.9)
    ax = fig.add_subplot(111)
    cm = plt.cm.get_cmap('RdYlBu')
    paths = ax.scatter(binom, gillespie, s = psi_means*300, c = counts, cmap = cm, alpha=0.35)
    cbar = fig.colorbar(paths, ax = ax)
#    cbar = ax.figure.colorbar(None, ax=ax)
    cbar.ax.set_ylabel("counts", rotation=-90, va="bottom")
    ax.set_xlabel("Binomial, std")
    ax.set_ylabel("Gillespie, std")
    
    ax.plot([0,np.max(binom)],[0,np.max(binom)], c="r")
    return 0
