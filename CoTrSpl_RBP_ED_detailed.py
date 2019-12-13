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


extended_model = True
RON_gene = True
sim_series = True
plot_3d_series = True

runtime = 60
init_mol_count = 100000

factor = 2

rbp_posistions = np.linspace(0, 700, 201)

if RON_gene:
    runtime = 50
    gene_len = 700
    vpol = 50
    
    u1_1_pos = 210
    u2_1_pos = 300
    u1_2_pos = 443
    u2_2_pos = 520
    u1_3_pos = 690
    
    #exon definition rates
    k1 = 5e-2
    k2 = 1e-5
    k3 = 1e-1
    
    k1_i = 0
    k2_i = 0
    k3_i = 0
    
    spl_r = 1
    ret_r = 0.01
    
    rbp_pos = 400
    rbp_bbr = 0.01 #basal binding rate
    rbp_br = 1 #pol2 associated binding rate
    rbp_e_up = 10
    rbp_e_down = 30
    rbp_h_c = 10
    
    pol_dist = 20 # nt's after pol pass the binding pos 
    
    
else: #experimental parameters
    runtime = 20
    gene_len = 700
    u1_1_pos = 210
    u2_1_pos = 300
    u1_2_pos = 443
    u2_2_pos = 520
    u1_3_pos = 690
    
    spl_i_tau = 4e-2
    spl_s_tau = 1e-1
    
    rbp_pos = 250
    rbp_radius = 40
    
    rbp_br = 1e-1
    rbp_ur = 1e-3
#    rbp_br = 20 * factor
#    rbp_ur = 0.0 * factor
    
    rbp_hill_c = 10
    
    vpol = 60

    
params = {"vpol": vpol,
        "gene_len": gene_len,
        "u1_1_pos": u1_1_pos , # U1 binding site position
        "u1_2_pos": u1_2_pos ,
        "u2_1_pos": u2_1_pos, # U2 bind. site pos 1
        "u2_2_pos": u2_2_pos,
        "u1_3_pos": u1_3_pos,
        
        "k1": k1_i,
        "k2": k2_i,
        "k3": k3_i,
        "k1_t": k1,
        "k2_t": k2,
        "k3_t": k3,
        
        "rbp_pos": rbp_pos,
        "rbp_bbr": rbp_bbr,
        "rbp_br": 0,
        "rbp_br_t": rbp_br,
        "rbp_e_up": rbp_e_up,
        "rbp_e_down": rbp_e_down,
        "rbp_h_c": rbp_h_c,
        "pol_dist": pol_dist,
        "spl": spl_r,
        "ret_r": 0,
        "ret_r_t": ret_r
        
        }
if(extended_model):
    s = bs.SimParam("CoTrSpl_RBP_extended", runtime, 10001, params = params,
                    init_state = {"P000": init_mol_count})
    
    for ext in ["", "_rbp"]:
        s.add_reaction("P000"+ext+"*k1", {"P100"+ext+"":1, "P000"+ext+"":-1}, "E1 def")
        s.add_reaction("P000"+ext+"*k2", {"P010"+ext+"":1, "P000"+ext+"":-1}, "E2 def")
        s.add_reaction("P000"+ext+"*k3", {"P001"+ext+"":1, "P000"+ext+"":-1}, "E3 def")
        s.add_reaction("P100"+ext+"*k2", {"P110"+ext+"":1, "P100"+ext+"":-1}, "E2 def")
        s.add_reaction("P100"+ext+"*k3", {"P101"+ext+"":1, "P100"+ext+"":-1}, "E3 def")
        s.add_reaction("P010"+ext+"*k1", {"P110"+ext+"":1, "P010"+ext+"":-1}, "E1 def")
        s.add_reaction("P010"+ext+"*k3", {"P011"+ext+"":1, "P010"+ext+"":-1}, "E3 def")
        s.add_reaction("P001"+ext+"*k1", {"P101"+ext+"":1, "P001"+ext+"":-1}, "E1 def")
        s.add_reaction("P001"+ext+"*k2", {"P011"+ext+"":1, "P001"+ext+"":-1}, "E2 def")
        
        s.add_reaction("P110"+ext+"*k3", {"P111"+ext+"":1, "P110"+ext+"":-1}, "E3 def")
        s.add_reaction("P101"+ext+"*k2", {"P111"+ext+"":1, "P101"+ext+"":-1}, "E2 def")
        s.add_reaction("P011"+ext+"*k1", {"P111"+ext+"":1, "P011"+ext+"":-1}, "E1 def")
    
    for spec in ["P000", "P001", "P010", "P100", "P011", "P101", "P110", "P111"]:
        spec_i = spec + "_rbp"
        s.add_reaction(spec + "*rbp_br", {spec:-1, spec_i:1}, "inh.")
        s.add_reaction(spec + "*ret_r", {spec:-1, "ret":1}, "retention")
        s.add_reaction(spec_i + "*ret_r", {spec_i:-1, "ret":1}, "retention")
    
    s.add_reaction("P111*spl", {"P111":-1, "Incl": 1}, "inclusion")
    s.add_reaction("P101*spl", {"P101":-1, "Skip": 1}, "skipping")
    
    s.add_reaction("P111_rbp*spl * (1-asym_porximity(rbp_pos, u1_1_pos, rbp_e_up, rbp_e_down, rbp_h_c)) \
                   * (1-asym_porximity(rbp_pos, u2_1_pos, rbp_e_up, rbp_e_down, rbp_h_c))\
                   * (1-asym_porximity(rbp_pos, u1_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))\
                   * (1-asym_porximity(rbp_pos, u2_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))",
                   {"P111_rbp":-1, "Incl":1, "TEST_INCL":1}, "inclusion")
    
    s.add_reaction("P101_rbp*spl * (1-asym_porximity(rbp_pos, u1_1_pos, rbp_e_up, rbp_e_down, rbp_h_c)) \
                   * (1-asym_porximity(rbp_pos, u2_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))",
                   {"P101_rbp":-1, "Skip":1, "TEST_SKIP":1}, "skipping")
    
    
    
else: #TODO
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

te1 = bs.TimeEvent("u1_1_pos/vpol", "k1=k1_t")
te2 = bs.TimeEvent("u1_2_pos/vpol", "k2=k2_t")
te3 = bs.TimeEvent("u1_3_pos/vpol", "k3=k3_t")
te4 = bs.TimeEvent("rbp_pos/vpol", "rbp_br=rbp_br_t")
te5 = bs.TimeEvent("(rbp_pos+pol_dist)/vpol", "rbp_br=rbp_bbr")
te6 = bs.TimeEvent("gene_len/vpol", "ret_r = ret_r_t")

s.add_timeEvent(te1)
s.add_timeEvent(te2)
s.add_timeEvent(te3)
s.add_timeEvent(te4)
s.add_timeEvent(te5)
s.add_timeEvent(te6)

s.compile_system()
#s.simulate_ODE = True
#s.simulate()
#s.plot_course()
#s.draw_pn(engine="dot", rates=False)
s.show_interface()
inh_curve = 'norm_proximity(t*vpol, rbp_pos, rbp_radius, rbp_hill_c)'
#s.plot_course(products= [inh_curve])

if sim_series:
    s.set_runtime(1e4)
    psis = []
    rets = []
    for i, rbp_pos in enumerate(rbp_posistions):
        print("Sim count: %d" % i)
        s.set_param("rbp_pos", rbp_pos)
        s.simulate()
        psi = s.get_psi_mean(ignore_fraction = 0.4)
        psis.append(psi)
        ret = s.get_res_col("ret")[-1]
        rets.append(ret/init_mol_count)
    
    fig, ax = plt.subplots()
    ax.plot(rbp_posistions, psis, lw=2, label = "PSI")
    ax.plot(rbp_posistions, rets, lw=1, label="ret")
    ax.set_xlabel("RBP pos")
    ax.set_ylabel("PSI")
    ax.set_title("u11: %d; u21: %d; u12: %d; u22: %d; Radius:(%d, %d)" % 
                 (u1_1_pos, u2_1_pos, u1_2_pos, u2_2_pos, rbp_e_up, rbp_e_down))
    ax.axvline(u1_1_pos, label="U11", linestyle="-.",lw =0.7)#, color = "red")
    ax.axvline(u2_1_pos, label="U21", linestyle="-.",lw =0.7)#, color = "red")
    ax.axvline(u1_2_pos, label="U12", linestyle="-.", lw =0.7)#, color = "red")
    ax.axvline(u2_2_pos, label="U22", linestyle="-.", lw =0.7)#, color = "red")
    ax.legend()
    ax2 = ax.twinx()
    inh_curve_u11 = [bs.asym_porximity(rbp_p, u1_1_pos, rbp_e_up, rbp_e_down, rbp_h_c) for rbp_p in rbp_posistions]
    ax2.plot(rbp_posistions,inh_curve_u11, label = "Inh. range on U11", linestyle=":", color="red")
    inh_curve_u22 = [bs.asym_porximity(rbp_p, u2_2_pos, rbp_e_up, rbp_e_down, rbp_h_c) for rbp_p in rbp_posistions]
    ax2.plot(rbp_posistions,inh_curve_u22, label = "Inh. range on U22", linestyle=":", color="green")
#    inh_curve_pos350 = [bs.norm_proximity(rbp_p,350 , rbp_radius, rbp_hill_c) for rbp_p in rbp_posistions]
#    ax2.plot(rbp_posistions,inh_curve_pos350, label = "Inh. range on 350", linestyle=":", color="orange")
    ax2.legend()

if plot_3d_series:

    s.set_runtime(1e4)
    vpols = np.linspace(1,400,100)
    vpols = np.logspace(1,3,100)
    rbp_poss = np.linspace(50, 650, 61)
    X, Y = np.meshgrid(rbp_poss, vpols)
    
    res = s.plot_par_var_2d(pars=dict(vpol=vpols, rbp_pos=rbp_poss),
                            func=s.get_psi_mean, ignore_fraction=0.9)
    
    Z = np.array(res.values)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,np.log10(Y),Z,  cmap=cm.coolwarm)
    ax.set_xlabel("RBP pos")
    ax.set_ylabel("vpol [nt/s]")
    ax.set_zlabel("PSI")
#    ax.set_yscale("log")
#    ax.yaxis._set_scale('log')
    
    step = 10
    indx_x = np.arange(0, len(rbp_poss)-1, step)
    indx_y = np.arange(0, len(vpols)-1, step)
    fig, ax  = plt.subplots()
    im = ax.imshow(Z)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("PSI", rotation=-90, va="bottom")
    ax.set_xticks(indx_x)
    ax.set_yticks(indx_y)
    # ... and label them with the respective list entries.
    col_labels = rbp_poss[indx_x]
    row_labels = vpols[indx_y]
    ax.set_xticklabels(col_labels, fontsize = 10, rotation=60)
    ax.set_yticklabels(row_labels, fontsize = 10)
   

