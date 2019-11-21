#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:20:54 2019

@author: timur
"""



import bioch_sim as bs
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np


extended_model = False
RON_gene = True
sim_series = False
plot_3d_series = True

runtime = 60
init_mol_count = 10000

factor = 1

if RON_gene:
    runtime = 20
    gene_len = 700
    u1_1_bs_pos = 210
    u2_1_bs_pos = 300
    u1_2_bs_pos = 443
    u2_2_bs_pos = 520
    
    spl_i_tau = 0.1 * factor
    spl_s_tau = 0.5 * factor
    
    rbp_pos = 250
    rbp_radius = 50
    
    
    rbp_br = 0.2 * factor
    rbp_ur = 0.02 * factor
#    rbp_br = 20 * factor
#    rbp_ur = 0.0 * factor
    
    rbp_hill_c = 15
    
    
    
    rbp_posistions = np.linspace(0, 650, 201)
    
    
    vpol = 60
    spl_r = 0.8
    
#    v1_1 = 0.8
#    v2_1 = 0.2
#    v1_2 = 0.1
#    v2_2 = 5

    
params = {"vpol": vpol,
        "u1_1_bs_pos": u1_1_bs_pos , # U1 binding site position
        "u1_2_bs_pos": u1_2_bs_pos ,
        "u2_1_bs_pos": u2_1_bs_pos, # U2 bind. site pos 1
        "u2_2_bs_pos": u2_2_bs_pos,
#        "u1_1_br": v1_1,  # binding rate of U1 
#        "u2_1_br": v2_1,
#        "u1_2_br": v1_2,  # binding rate of U1
#        "u2_2_br": v2_2,
#        "u1ur": 0.001,  # unbinding rate of U1 
#        "u2ur": 0.001, # unbinding rate of U1
        "spl_i_tau": spl_i_tau, #splicing rate of introns, without exons
        "spl_i1": 0, #splicing rate of first intron
        "spl_i2": 0,
        "spl_s": 0, # skipping splicing rate, with exon
        "spl_s_tau":spl_s_tau,
        "rbp_pos": rbp_pos,
        "rbp_radius": rbp_radius,
        "rbp_hill_c": rbp_hill_c,
        "rbp_br": 0,
        "rbp_br_tau": rbp_br,
        "rbp_ur": rbp_ur,
#                "tr_term_rate": 100,
#                "s1":s1, "s2":s2, "s3": 1e-4,
        # http://book.bionumbers.org/how-fast-do-rnas-and-proteins-degrade/
        "d1": 2e-4, "d2":2e-4, "d3":1e-3 # mRNA half life: 10-20 h -> lambda: math.log(2)/hl
        }
if(extended_model):
    s = bs.SimParam("CoTrSpl_RBP_extended", runtime, 10001, params = params,
                    init_state = {"mRNA": init_mol_count})
    
    s.add_reaction("mRNA*rbp_br", dict(mRNA=-1, mRNAinh=1), "RBP to mRNA")
    s.add_reaction("mRNAinh * rbp_ur", dict(mRNA=1, mRNAinh=-1), "RBP from mRNA")
    s.add_reaction("mRNA*spl_i1", dict(mRNA=-1, ret_i2=1), "1. Intr splicing")
    s.add_reaction("mRNA*spl_i2", dict(mRNA=-1, ret_i1=1), "2. Intr splicing")
    s.add_reaction("mRNA*spl_s", dict(mRNA=-1, Skip=1), "Skipping")
    
    s.add_reaction("ret_i1 * spl_i1", dict(ret_i1=-1, Incl=1))
    s.add_reaction("ret_i2 * spl_i2", dict(ret_i2=-1, Incl=1))
    s.add_reaction("ret_i1 * rbp_br", dict(ret_i1=-1, ret_i1_inh=1))
    s.add_reaction("ret_i1_inh * rbp_ur", dict(ret_i1_inh=-1, ret_i1=1))
    s.add_reaction("ret_i2 * rbp_br", dict(ret_i2=-1, ret_i2_inh=1))
    s.add_reaction("ret_i2_inh * rbp_ur", dict(ret_i2_inh=-1, ret_i2=1))
    
    s.add_reaction("mRNAinh*spl_i1 * (1-norm_proximity(rbp_pos, u1_1_bs_pos, rbp_radius, rbp_hill_c)) \
     * (1-norm_proximity(rbp_pos, u2_1_bs_pos, rbp_radius, rbp_hill_c))",
        dict(mRNAinh=-1, ret_i2_inh=1), "1. Intr splicing")
    s.add_reaction("mRNAinh*spl_i2 * (1-norm_proximity(rbp_pos, u1_2_bs_pos, rbp_radius, rbp_hill_c)) \
     * (1-norm_proximity(rbp_pos, u2_2_bs_pos, rbp_radius, rbp_hill_c))",
        dict(mRNAinh=-1, ret_i1_inh=1), "2. Intr splicing")
    s.add_reaction("mRNAinh*spl_s * (1-norm_proximity(rbp_pos, u1_1_bs_pos, rbp_radius, rbp_hill_c)) \
     * (1-norm_proximity(rbp_pos, u2_2_bs_pos, rbp_radius, rbp_hill_c))",
        dict(mRNAinh=-1, Skip=1), "Skipping")
    s.add_reaction("ret_i2_inh*spl_i2 * (1-norm_proximity(rbp_pos, u1_2_bs_pos, rbp_radius, rbp_hill_c)) \
     * (1-norm_proximity(rbp_pos, u2_2_bs_pos, rbp_radius, rbp_hill_c))",
        dict(ret_i2_inh=-1, Incl=1), "Inclusion")
    s.add_reaction("ret_i1_inh*spl_i1 * (1-norm_proximity(rbp_pos, u1_1_bs_pos, rbp_radius, rbp_hill_c)) \
     * (1-norm_proximity(rbp_pos, u2_1_bs_pos, rbp_radius, rbp_hill_c))",
        dict(ret_i1_inh=-1, Incl=1), "Inclusion")
else:
    s = bs.SimParam("CoTrSpl_RBP_simple", runtime, 10001, params = params,
                    init_state = {"mRNA": init_mol_count})
    
    s.add_reaction("mRNA*rbp_br", dict(mRNA=-1, mRNAinh=1), "RBP to mRNA")
    s.add_reaction("mRNAinh * rbp_ur", dict(mRNA=1, mRNAinh=-1), "RBP from mRNA")
    s.add_reaction("mRNA*spl_i1", dict(mRNA=-1, Incl=1), "1. Intr splicing")
    s.add_reaction("mRNA*spl_i2", dict(mRNA=-1, Incl=1), "2. Intr splicing")
    s.add_reaction("mRNA*spl_s", dict(mRNA=-1, Skip=1), "Skipping")
    
    s.add_reaction("mRNAinh*spl_i1 * (1-norm_proximity(rbp_pos, u1_1_bs_pos, rbp_radius, rbp_hill_c)) \
     * (1-norm_proximity(rbp_pos, u2_1_bs_pos, rbp_radius, rbp_hill_c))",
        dict(mRNAinh=-1, Incl=1), "1. Intr splicing")
    s.add_reaction("mRNAinh*spl_i2 * (1-norm_proximity(rbp_pos, u1_2_bs_pos, rbp_radius, rbp_hill_c)) \
     * (1-norm_proximity(rbp_pos, u2_2_bs_pos, rbp_radius, rbp_hill_c))",
        dict(mRNAinh=-1, Incl=1), "2. Intr splicing")
    s.add_reaction("mRNAinh*spl_s * (1-norm_proximity(rbp_pos, u1_1_bs_pos, rbp_radius, rbp_hill_c)) \
     * (1-norm_proximity(rbp_pos, u2_2_bs_pos, rbp_radius, rbp_hill_c))",
        dict(mRNAinh=-1, Skip=1), "Skipping")

te1 = bs.TimeEvent("u2_1_bs_pos/vpol", "spl_i1=spl_i_tau")
te2 = bs.TimeEvent("u2_2_bs_pos/vpol", "spl_i2=spl_i_tau; spl_s=spl_s_tau")
te3 = bs.TimeEvent("rbp_pos/vpol", "rbp_br=rbp_br_tau")

s.add_timeEvent(te1)
s.add_timeEvent(te2)
s.add_timeEvent(te3)

s.compile_system()
#s.simulate_ODE = True
#s.simulate()
#s.plot_course()
#s.draw_pn(engine="dot", rates=False)
s.show_interface()
inh_curve = 'norm_proximity(t*vpol, rbp_pos, rbp_radius, rbp_hill_c)'
#s.plot_course(products= [inh_curve])

if sim_series:
    s.set_runtime(1e3)
    psis = []
    for i, rbp_pos in enumerate(rbp_posistions):
        print("Sim count: %d" % i)
        s.set_param("rbp_pos", rbp_pos)
        s.simulate()
        print()
        psi = s.get_psi_mean(ignore_fraction = 0.4)
        psis.append(psi)
    
    fig, ax = plt.subplots()
    ax.plot(rbp_posistions, psis)
    ax.set_xlabel("RBP pos")
    ax.set_ylabel("PSI")
    ax.set_title("u11: %d; u21: %d; u12: %d; u22: %d; Radius: %d" % 
                 (u1_1_bs_pos, u2_1_bs_pos, u1_2_bs_pos, u2_2_bs_pos, rbp_radius))
    ax.axvline(u1_1_bs_pos, label="U11", linestyle=".", color = "red")
    ax.axvline(u2_1_bs_pos, label="U21", linestyle=".", color = "red")
    ax.axvline(u1_2_bs_pos, label="U12", linestyle=".", color = "red")
    ax.axvline(u2_2_bs_pos, label="U22", linestyle=".", color = "red")


if plot_3d_series:

    s.set_runtime(1e4)
    vpols = np.linspace(1,100,100)
    rbp_poss = np.linspace(50, 650, 120)
    X, Y = np.meshgrid(rbp_poss, vpols)
    
    res = s.plot_par_var_2d(pars=dict(vpol=vpols, rbp_pos=rbp_poss),
                            func=s.get_psi_mean, ignore_fraction=0.9)
    
    Z = np.array(res.values)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z,  cmap=cm.coolwarm)
    ax.set_xlabel("RBP pos")
    ax.set_ylabel("vpol [nt/s]")
    ax.set_zlabel("PSI")
    
    
    fig, ax  = plt.subplots()
    ax.imshow(Z)
   

