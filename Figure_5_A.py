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
from axes_zoom_effect import *
import scipy.stats as st

sim_count = 100
init_mol_count = 100


fig = "5A"


if fig == "5B":
    s = bs.coTrSplMechanistic()
    init_sp = "P000"
    s.set_color("P100", "magenta")
    s.set_color("P000_inh", "orange")
    s.set_param("rbp_pos", 330)
    s.params.update( dict(vpol=50,
            rbp_pos = 480,
            rbp_br_t=5,
            rbp_inh=0.99,
            k1_t = 1,
            k2_t = 0.1,
            k3_t = 1,
            ret_t = 1e-3,
            rbp_e_up = 30,
            rbp_e_down = 40
            ))
    prod_to_show = ("Incl", "Skip",  "P000", "P100",  "P100_inh")
    runtime = 50
    start_segment = (0,20)
    s.set_raster(1000)
    x_label = "time after\ninitiation [s]"
    title="Mechanistic exon\ndefinition model"
    noise_pars = dict(par="rbp_pos", vals=np.linspace(220,280,20),
                          init_sp="P000", init_mol_counts=[10,20,30,40],
                          s_count= 2000, mc_variable = True)
    file_name = "Figure_5B.svg"
    models = [s]

if fig == "5A":
    #model corresponds to Fig 3 U-shape
    s = bs.coTrSplCommitment(l=8, m1=2, m2=3, k=0, n=2,
                    ki= 5e-2, ks=5e-3, kesc=0.5)["td_m"]
    init_sp = "mRNA"
    s.set_color("mRNAinh", "orange")
    prod_to_show = ["mRNA", "mRNAinh", "Incl", "Skip"]
    runtime = 1000
    start_segment = (0,5)
    s.set_raster(10000)
    x_label = "time after exon 2\n synthesis [s]"
    title="Splicing commitment\nmodel"
    #s shape
    s2 =  bs.coTrSplCommitment(l=8, m1=0, m2=1, k=0, n=2,
                    ki= 0.1, ks=0.02, kesc=1)["td_m"]
    #bell shape
    s3 =  bs.coTrSplCommitment(l=8, m1=0, m2=1, k=0, n=2,
                    ki= 1e-1, ks=2e-1, kesc=0.2)["td_m"]
    
    noise_pars = dict(par="vpol", vals=np.logspace(0,3,15),
                          init_sp="mRNA", init_mol_counts=[10,50,200,1000],
                          s_count= 5000, mc_variable = False)
    file_name = "Figure_5A.svg"
    models =[s, s2, s3]
    
figsize=(6,6)

s.set_color("Incl", "green")
s.set_color("Skip", "red")

raster_start = np.arange(0, start_segment[1], 100)
raster_after_s = np.arange(start_segment[1], s.runtime, 100)

s.raster = np.concatenate((raster_start,raster_after_s))

s.set_runtime(runtime)
s.set_init_species(init_sp, init_mol_count)
s.simulate(sim_count, verbose = False)

fig = plt.figure(figsize=figsize, dpi=100)
ax_whole = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1, fig=fig)
ax_start = plt.subplot2grid((2,2), (1,0), colspan=2, rowspan=1, fig=fig)
ax_std = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1, fig=fig)


#s.plot_series(products=("Incl", "Skip",  "P000", "P001", "P110_inh", "P011_inh", "P010_inh", "P111", "P111_inh", "ret" ))
ax_whole = s.plot_series(products=prod_to_show, ax=ax_whole, scale=0.7)
ax_start = s.plot_series(products=prod_to_show, t_bounds=start_segment, ax=ax_start, scale=0.7)
ax_whole.set_title(title, weight="bold")
ax_start.set_title(None)
ax_whole.set_xlabel(x_label, weight="bold")
ax_start.set_xlabel(None)
ax_start.set_ylabel("Molecule count", weight="bold")
ax_whole.set_ylabel("Molecule count", weight="bold")
ax_start.legend(fontsize=9)

ax_std.set_title(None)

#https://matplotlib.org/3.1.0/api/_as_gen/mpl_toolkits.axes_grid1.inset_locator.BboxConnector.html
line_kw = dict(ls="--", color="grey" )
#https://matplotlib.org/examples/pylab_examples/axes_zoom_effect.html
zoom_effect01(ax_whole, ax_start, start_segment[0], start_segment[1],
              prop_patches=dict(color="#b1f6f5"),
              **line_kw)

#https://matplotlib.org/api/text_api.html#matplotlib.text.Text
text_kw = dict(fontsize=10)
s.annotate_timeEvents2(ax_start, y_axes_offset=-0.1, text_ax_offset= -0.5,
                       line_kw = line_kw, text_kw = text_kw)

#plt.savefig("FIGURE_5.jpeg", dpi=500)

#s.plot_series()
#s.plot_course(products=["Incl", "Skip",  "P000", "P001" ])
#s.plot_parameters()
#s.set_runtime(1e4)
#
#s.simulate(sim_count, verbose = False)
#res = s.get_res_by_expr_2("Incl/(Incl+Skip)", t_bounds = (s.get_runtime()*0.999,np.inf), series=True )
#res = np.array(res)
#fig, ax = plt.subplots()
##ax.plot(res.T, c="blue", lw=0.2)
#ax.hist(res[:,-1])
#ax.set_title("PSI distribution")
#ax.set_xlabel("PSI")


s.set_runtime(runtime)

def compare_to_binomial(models, par="vpol", vals=np.linspace(10,1000,50), init_sp = "mRNA",
                        init_mol_counts = [20,200,500], s_count = 100, mc_variable=False, ax = None):
    
    if type(models) is not list:
        models = [models]
    
    psi_means = np.zeros(len(vals) * len(init_mol_counts) * len(models))
    psi_stds = np.array(psi_means)
    counts = np.array(psi_means)
    m_is  = np.array(psi_means)
    i=0
    m_i = 0
    for s in models:
        for m_count in init_mol_counts:
            s.set_init_species(init_sp, m_count)
            for  v in vals:
                print("mc: ", m_count, "\n", par, ": ", v)
                s.set_param(par, v)
                s.simulate(s_count, verbose=False)
                incls = s.get_res_col("Incl", series=True)[:,-1]
                skips = s.get_res_col("Skip", series=True)[:,-1]
                counts[i] = np.mean(skips + incls)
                
                psis_end = incls/(incls+skips)
                psi_means[i] = np.mean(psis_end)
                psi_stds[i] = np.std(psis_end)
                m_is[i] = m_i
                i += 1
        m_i += 1
        
    if(ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    
    if mc_variable:
        binom = sp.stats.binom.std(counts, psi_means)/counts
        cmap = plt.cm.get_cmap('RdYlBu')
        paths = ax.scatter(psi_stds, binom, s = (psi_means+0.5)*100, c = counts, cmap = cmap, alpha=0.35)
        cbar = plt.colorbar(paths, ax = ax)
    #    cbar = ax.figure.colorbar(None, ax=ax)
        cbar.ax.set_ylabel("Skip + Incl", rotation=-90, va="bottom")
        ax.set_ylabel("std(PSI), Binomial", weight="bold")
        ax.set_xlabel("std(PSI), Gillespie", weight="bold")
#        ax.set_title(str(s_count) + " simulations per point")
        ax.set_title("Comparison to bonomial\ndistribution", weight="bold")
        ax.plot([0,np.max(binom)],[0,np.max(binom)], c="r")
    else:
        cmap = plt.cm.get_cmap('RdYlBu')
        psis_th = np.linspace(0,1,100)
        for mc, ls in zip(init_mol_counts, ["-","--","-.",":"]):
            b_stds = [st.binom.std(mc, p)/mc for p in psis_th]
            ax.plot(psis_th, b_stds, color = "black", lw=1,ls=ls,
                    alpha = 0.5,
                    label = "mc: %d" % mc )
        paths = ax.scatter(psi_means, psi_stds, s = 40, c = m_is, edgecolors = "black",
                           cmap = cmap, alpha=0.35)
        ax.set_xlabel("mean(PSI)", weight="bold")
        ax.set_ylabel("std(PSI)", weight="bold")
        ax.set_title("Noise-mean relationship", weight="bold")
        ax.legend(ncol=1, fontsize=6)
#                  bbox_to_anchor =(1.1,0,1, 0.1))
        
    return dict(counts = counts, psi_means= psi_means, psi_stds = psi_stds,ax= ax, fig=fig)

res = compare_to_binomial(models, ax = ax_std, **noise_pars)

fig.tight_layout()
plt.savefig(file_name, dpi=300)
