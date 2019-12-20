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
sim_series = False
plot_3d_series_stoch = False
plot_3d_series_det = True
plot_3d_series_rbp_br_titr = False



spl_inh= False
runtime = 60
init_mol_count = 1000

factor = 4

rbp_posistions = np.linspace(0, 700, 201)

if RON_gene:
    runtime = 20
    gene_len = 700
    vpol = 50
    
    u1_1_pos = 210
    u2_1_pos = 300
    u1_2_pos = 443
    u2_2_pos = 520
    u1_3_pos = 690
    
    #exon definition rates
    k1 = 5e-2 * factor
    k2 = 2e-2 * factor
    k3 = 1 * factor
    
    k1_i = 0
    k2_i = 0
    k3_i = 0
    
    k1_inh = "k1_t * (1-rbp_inh*asym_porximity(rbp_pos, u1_1_pos, rbp_e_up, rbp_e_down, rbp_h_c))"
    k2_inh = "k2_t * (1-rbp_inh*asym_porximity(rbp_pos, u2_1_pos, rbp_e_up, rbp_e_down, rbp_h_c)) \
                   * (1-rbp_inh*asym_porximity(rbp_pos, u1_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))"
    k3_inh = "k3_t * (1-rbp_inh*asym_porximity(rbp_pos, u1_3_pos, rbp_e_up, rbp_e_down, rbp_h_c)) \
                   * (1-rbp_inh*asym_porximity(rbp_pos, u2_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))"
    
    
    spl_r = 0.1 * factor
    ret_r = 0.001 * factor
    
    rbp_pos = 350
    rbp_inh = 0.98
    rbp_bbr = 1e-9 #basal binding rate
    rbp_br = 26*k2 #pol2 associated binding rate
    rbp_br = 0.6 #pol2 associated binding rate
    rbp_e_up = 30
    rbp_e_down = 50
    rbp_h_c = 6
    
    pol_dist = 10 # max nt's after pol can bring somth. to nascRNA
    
    
else: #experimental parameters
    runtime = 20
    gene_len = 700
    vpol = 50
    
    u1_1_pos = 210
    u2_1_pos = 300
    u1_2_pos = 443
    u2_2_pos = 520
    u1_3_pos = 690
    
    #exon definition rates
    k1 = 5e-2 * factor
    k2 = 2e-2 * factor
    k3 = 1 * factor
    
    k1_i = 0
    k2_i = 0
    k3_i = 0
    
    k1_inh = "k1_t * (1-rbp_inh*asym_porximity(rbp_pos, u1_1_pos, rbp_e_up, rbp_e_down, rbp_h_c))"
    k2_inh = "k2_t * (1-rbp_inh*asym_porximity(rbp_pos, u2_1_pos, rbp_e_up, rbp_e_down, rbp_h_c)) \
                   * (1-rbp_inh*asym_porximity(rbp_pos, u1_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))"
    k3_inh = "k3_t * (1-rbp_inh*asym_porximity(rbp_pos, u1_3_pos, rbp_e_up, rbp_e_down, rbp_h_c)) \
                   * (1-rbp_inh*asym_porximity(rbp_pos, u2_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))"
    
    
    spl_r = 0.1 * factor
    ret_r = 0.001 * factor
    
    rbp_pos = 350
    rbp_inh = 0.98
    rbp_bbr = 1e-9 #basal binding rate
    rbp_br = 26*k2 #pol2 associated binding rate
    rbp_br = 0.6 #pol2 associated binding rate
    rbp_e_up = 30
    rbp_e_down = 50
    rbp_h_c = 6
    
    pol_dist = 10 # max nt's after pol can bring somth. to nascRNA

    
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
        
        "k1_inh": 0,
        "k2_inh": 0,
        "k3_inh": 0,
        
        "k1_inh_t": k1_inh,
        "k2_inh_t": k2_inh,
        "k3_inh_t": k3_inh,
        
        
        "rbp_pos": rbp_pos,
        "rbp_bbr": rbp_bbr,
        "rbp_br": 0,
        "rbp_br_t": rbp_br,
        "rbp_e_up": rbp_e_up,
        "rbp_e_down": rbp_e_down,
        "rbp_h_c": rbp_h_c,
        "rbp_inh": rbp_inh,
        "pol_dist": pol_dist,
        "spl_i": spl_r,
        "spl_s": spl_r*0.5,
        "ret_r": 0,
        "ret_r_t": ret_r
        
        }
if(extended_model):
    s = bs.SimParam("CoTrSpl_RBP_extended", runtime, 10001, params = params,
                    init_state = {"P000": init_mol_count})
    
    for ext in ["", "_inh"]:
        s.add_reaction("P000"+ext+"*k1"+ext, {"P100"+ext+"":1, "P000"+ext+"":-1}, "E1 def")
        s.add_reaction("P000"+ext+"*k2"+ext, {"P010"+ext+"":1, "P000"+ext+"":-1}, "E2 def")
        s.add_reaction("P000"+ext+"*k3"+ext, {"P001"+ext+"":1, "P000"+ext+"":-1}, "E3 def")
        s.add_reaction("P100"+ext+"*k2"+ext, {"P110"+ext+"":1, "P100"+ext+"":-1}, "E2 def")
        s.add_reaction("P100"+ext+"*k3"+ext, {"P101"+ext+"":1, "P100"+ext+"":-1}, "E3 def")
        s.add_reaction("P010"+ext+"*k1"+ext, {"P110"+ext+"":1, "P010"+ext+"":-1}, "E1 def")
        s.add_reaction("P010"+ext+"*k3"+ext, {"P011"+ext+"":1, "P010"+ext+"":-1}, "E3 def")
        s.add_reaction("P001"+ext+"*k1"+ext, {"P101"+ext+"":1, "P001"+ext+"":-1}, "E1 def")
        s.add_reaction("P001"+ext+"*k2"+ext, {"P011"+ext+"":1, "P001"+ext+"":-1}, "E2 def")
        
        s.add_reaction("P110"+ext+"*k3"+ext, {"P111"+ext+"":1, "P110"+ext+"":-1}, "E3 def")
        s.add_reaction("P101"+ext+"*k2"+ext, {"P111"+ext+"":1, "P101"+ext+"":-1}, "E2 def")
        s.add_reaction("P011"+ext+"*k1"+ext, {"P111"+ext+"":1, "P011"+ext+"":-1}, "E1 def")
    
    for spec in ["P000", "P001", "P010", "P100", "P011", "P101", "P110", "P111"]:
        spec_i = spec + "_inh"
        s.add_reaction(spec + "*rbp_br", {spec:-1, spec_i:1}, "inh.")
        s.add_reaction(spec + "*ret_r", {spec:-1, "ret":1}, "retention")
        s.add_reaction(spec_i + "*ret_r", {spec_i:-1, "ret":1}, "retention")
    
    s.add_reaction("P111*spl_i", {"P111":-1, "Incl": 1}, "inclusion")
    s.add_reaction("P101*spl_s", {"P101":-1, "Skip": 1}, "skipping")
#    s.add_reaction("P111*spl_s", {"P111":-1, "Skip": 1}, "skipping")
    
    if spl_inh:
        s.add_reaction("P111_inh*spl_i * (1-rbp_inh*asym_porximity(rbp_pos, u1_1_pos, rbp_e_up, rbp_e_down, rbp_h_c)) \
                       * (1-rbp_inh*asym_porximity(rbp_pos, u2_1_pos, rbp_e_up, rbp_e_down, rbp_h_c))\
                       * (1-rbp_inh*asym_porximity(rbp_pos, u1_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))\
                       * (1-rbp_inh*asym_porximity(rbp_pos, u2_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))",
                       {"P111_inh":-1, "Incl":1, "TEST_INCL":1}, "inclusion")
        
        s.add_reaction("P101_inh*spl_s * (1-rbp_inh*asym_porximity(rbp_pos, u1_1_pos, rbp_e_up, rbp_e_down, rbp_h_c)) \
                       * (1-rbp_inh*asym_porximity(rbp_pos, u2_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))",
                       {"P101_inh":-1, "Skip":1, "TEST_SKIP":1}, "skipping")
        s.add_reaction("P111_inh*spl_s * (1-rbp_inh*asym_porximity(rbp_pos, u1_1_pos, rbp_e_up, rbp_e_down, rbp_h_c)) \
                       * (1-rbp_inh*asym_porximity(rbp_pos, u2_2_pos, rbp_e_up, rbp_e_down, rbp_h_c))",
                       {"P111_inh":-1, "Skip":1, "TEST_SKIP":1}, "skipping")
    else:
        s.add_reaction("P111_inh*spl_i",
                       {"P111_inh":-1, "Incl":1, "TEST_INCL":1}, "inclusion")
        
        s.add_reaction("P101_inh*spl_s",
                       {"P101_inh":-1, "Skip":1, "TEST_SKIP":1}, "skipping")
#        s.add_reaction("P111_inh*spl_s",
#                       {"P111_inh":-1, "Skip":1, "TEST_SKIP":1}, "skipping")
        
    
    
else: #TODO
    pass

te1 = bs.TimeEvent("u1_1_pos/vpol", "k1=k1_t; k1_inh=k1_inh_t", name="Ex1 avail")
te2 = bs.TimeEvent("u1_2_pos/vpol", "k2=k2_t; k2_inh=k2_inh_t", name="Ex2 avail")
te3 = bs.TimeEvent("u1_3_pos/vpol", "k3=k3_t; k3_inh=k3_inh_t", name="Ex3 avail")
te4 = bs.TimeEvent("rbp_pos/vpol", "rbp_br=rbp_br_t", name="RBP b. start")
te5 = bs.TimeEvent("(rbp_pos+pol_dist)/vpol", "rbp_br=rbp_bbr", name="RBP b. end")
te6 = bs.TimeEvent("gene_len/vpol", "ret_r = ret_r_t", name="ret possible")
#
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

s.plot_parameters(parnames=["k1","k2","k3","k1_inh","k2_inh","k3_inh", "ret_r"],
                  parnames2=["rbp_br"])

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
    ax.plot(rbp_posistions, rets, lw=1, label="ret %")
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

if plot_3d_series_stoch:

    s.set_runtime(1e4)
    vpols = np.linspace(1,400,100)
    vpols = np.logspace(0,3,100)
    rbp_poss = np.linspace(50, 650, 61)
    X, Y = np.meshgrid(rbp_poss, np.log10(vpols))
    
    res = s.plot_par_var_2d(pars=dict(vpol=vpols, rbp_pos=rbp_poss),
                            func=s.get_psi_mean, ignore_fraction=0.9)
    
    Z = np.array(res.values)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z,  cmap=cm.coolwarm)
    ax.set_xlabel("RBP pos")
    ax.set_ylabel("log10 (vpol [nt/s])")    
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
   

if plot_3d_series_det:

    s.set_runtime(1e5)
    s.set_raster(1001)
    vpols = np.linspace(1,400,100)
    vpols = np.logspace(0,3,50)
    rbp_poss = np.linspace(425, 435, 31)
    rbp_poss = np.linspace(150, 350, 41)
    X, Y = np.meshgrid(rbp_poss, np.log10(vpols))
    
    Z = np.zeros((len(vpols), len(rbp_poss)))
    rets = np.array(Z)
    
    for i, vpol in enumerate(vpols):
        s.set_param("vpol", vpol)
        for j, rbp_p in enumerate(rbp_poss):
            s.set_param("rbp_pos", rbp_p)
            print(vpol, rbp_p)
            
            res = s.simulate(stoch=False, ODE=True)
            incl = s.get_res_col("Incl", method="ODE")[-1]
            skip = s.get_res_col("Skip", method="ODE")[-1]
            ret = s.get_res_col("ret", method="ODE")[-1]
            if(ret<0 or incl <0 or skip <0):
                print("AAAAAAAAAAAAAAA\n", "vpol:", vpol, "rbp_pos", rbp_p)
                print(ret, incl, skip)
            psi = incl/(incl+skip)
            ret_perc = ret/init_mol_count
            Z[i,j] = psi
            rets[i,j] = ret_perc
            
            
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z,  cmap=cm.coolwarm)
    ax.set_xlabel("RBP pos")
    ax.set_ylabel("log10 (vpol [nt/s])")    
    ax.set_zlabel("PSI")
#    ax.set_yscale("log")
#    ax.yaxis._set_scale('log')
    
    step = 10
    indx_x = np.arange(0, len(rbp_poss)-1, step)
    indx_y = np.arange(0, len(vpols)-1, step)
    fig, ax  = plt.subplots()
    im = ax.imshow(rets)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("ret %", rotation=-90, va="bottom")
    ax.set_xticks(indx_x)
    ax.set_yticks(indx_y)
    # ... and label them with the respective list entries.
    col_labels = rbp_poss[indx_x]
    row_labels = vpols[indx_y]
    ax.set_xticklabels(col_labels, fontsize = 10, rotation=60)
    ax.set_yticklabels(row_labels, fontsize = 10)
    ax.set_xlabel("RBP pos")
    ax.set_ylabel("vpol")

if plot_3d_series_rbp_br_titr:

    s.set_runtime(1e5)
    s.set_raster(1001)
    s.set_param("rbp_pos", 310)
    vpols = np.linspace(1,400,100)
    vpols = np.logspace(0,3,50)
    rbp_brs = np.linspace(0.01, 0.5, 50)
    X, Y = np.meshgrid(rbp_brs, np.log10(vpols))
    
    Z = np.zeros((len(vpols), len(rbp_brs)))
    rets = np.array(Z)
    
    for i, vpol in enumerate(vpols):
        s.set_param("vpol", vpol)
        for j, rbp_br in enumerate(rbp_brs):
            s.set_param("rbp_br_t", rbp_br)
            res = s.simulate(stoch=False, ODE=True)
            incl = s.get_res_col("Incl", method="ODE")[-1]
            skip = s.get_res_col("Skip", method="ODE")[-1]
            ret = s.get_res_col("ret", method="ODE")[-1]
            psi = incl/(incl+skip)
            ret_perc = ret/init_mol_count
            Z[i,j] = psi
            rets[i,j] = ret_perc
            
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z,  cmap=cm.coolwarm)
    ax.set_xlabel("RBP binding rate")
    ax.set_ylabel("log10 (vpol [nt/s])")    
    ax.set_zlabel("PSI")
#    ax.set_yscale("log")
#    ax.yaxis._set_scale('log')
    
    step = 10
    indx_x = np.arange(0, len(rbp_brs)-1, step)
    indx_y = np.arange(0, len(vpols)-1, step)
    fig, ax  = plt.subplots()
    im = ax.imshow(rets)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("ret %", rotation=-90, va="bottom")
    ax.set_xticks(indx_x)
    ax.set_yticks(indx_y)
    # ... and label them with the respective list entries.
    col_labels = rbp_brs[indx_x]
    row_labels = vpols[indx_y]
    ax.set_xticklabels(col_labels, fontsize = 10, rotation=60)
    ax.set_yticklabels(row_labels, fontsize = 10)
    ax.set_xlabel("RBP binding rate")
    ax.set_ylabel("vpol")

if False:
    vpol = 1.3257113655901092
    rbp_p = 325
    s.set_param("vpol", vpol)
    s.set_param("rbp_pos", rbp_p)
    s.simulate_ODE = True
    s.simulate(ODE=True)
    s.show_interface()
