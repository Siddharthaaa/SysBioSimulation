#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:39:26 2019

@author: timur

Modelling cotranscriptional splicing
    - discontinuous functions approach

"""


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import scipy.stats as st
import pyabc as pa
import pandas as pd
import support_th as sth
import bioch_sim as bs
import glob

#parameters 

gene_len = 100
u1_bs_pos = 10
u2_bs_pos_1 = 30
u2_bs_pos_2 = 60


params = {"pr_on": 2, "pr_off" : 0.1,
        "elong_v": 100,
        "gene_len": gene_len,
        "u1_bs_pos": u1_bs_pos , # U1 binding site position
        "u1_br_p1": 1,  # binding rate of U1
        "u1_ur_p1": 0.02, # unbinding rate of U1
        "u1_br_p2": 1,  # binding rate of U1
        "u1_ur_p2": 0.02, # unbinding rate of U1
        "u2_bs_pos_1": u2_bs_pos_1, # U2 bind. site pos 1
        "u2_bs_pos_2": u2_bs_pos_2,
        "u2_br_p1": 1,
        "u2_br_p2": 1,
        "u2_ur_p1": 0.1,
        "u2_ur_p2": 0.11,
        "tr_term_rate": 1000,
        "s1":1, "s2":1, "s3": 0.1,
        "d0":0.01, "d1": 0.01, "d2":0.01, "d3":0.1
        }


s = bs.SimParam("Cotranscriptional splicing", 10000, 10001, params = params,
                init_state = {"Pol_on":0, "Pol_off": 1,
                              "nascRNA_bc": 0,
                              "Pol_pos": 0,
                              "Skip":0, "Incl": 0, "ret": 0,
                              "Skip_nasc":0, "Skip_nasc": 0,
                              "ret_11":0, "ret_01":0, "ret_10":0,
                              "SS_ON":0, # splicing sites are free (splicing possible)
                              "U1_1":0, "U1_2":0,   #Splicosome units U1 binded
                              "U2_1":0, "U2_2":0,
                              "Intr1":0, "Intr2":0, "Exon1":0})

s.add_reaction("pr_on * Pol_off", {"Pol_on":1, "Pol_off": -1, "SS_ON": 1, "Intr1":1,"Intr2":1, "Exon1":1} )

s.add_reaction("elong_v * Pol_on if Pol_pos < gene_len else 0", {"nascRNA_bc": 1, "Pol_pos":1})
s.add_reaction("tr_term_rate * SS_ON if Pol_pos >= gene_len else 0", {"mRNA":1, "Pol_pos": -gene_len,
                                                      "nascRNA_bc": -gene_len,
                                                      "Pol_off":1, "Pol_on":-1,
                                                      "SS_ON":-1})
s.add_reaction("u1_br * SS_ON if Pol_pos > u1_bs_pos and U1 < 1  else 0", {"U1":1 } )
s.add_reaction("u1_ur if U1 > 0  else 0", {"U1":-1} )
s.add_reaction("u2_br_p1*SS_ON if Pol_pos > u2_bs_pos_1 and U2_1 < 1  else 0", {"U2_1":1 } )
s.add_reaction("u2_ur_p1 * U2_1", {"U2_1":-1 })
s.add_reaction("u2_br_p2*SS_ON if Pol_pos > u2_bs_pos_2 and U2_2 < 1  else 0", {"U2_2":1 } )
s.add_reaction("u2_ur_p2 * U2_2", {"U2_2":-1 })
s.add_reaction("U1 * U2_1 * SS_ON * s1", {"Incl_nasc":1, "U1": -1, "U2_1": -1, "SS_ON":-1})#,"nascRNA_bc": -(u2_bs_pos_1 - u1_bs_pos) })
s.add_reaction("tr_term_rate * Incl_nasc if Pol_pos >= gene_len else 0", {"Incl":1, "Incl_nasc":-1,
                                                        "Pol_pos": -gene_len,
                                                      "nascRNA_bc": -gene_len,
                                                      "Pol_off":1, "Pol_on":-1})
s.add_reaction("U1 * U2_2 * SS_ON * s2", {"Skip_nasc":1, "U1": -1, "U2_2": -1, "SS_ON":-1})#,  "nascRNA_bc": -(u2_bs_pos_2 - u1_bs_pos) })
s.add_reaction("tr_term_rate * Skip_nasc if Pol_pos >= gene_len else 0", {"Skip":1, "Skip_nasc":-1,
                                                                          "Pol_pos": -gene_len,
                                                      "nascRNA_bc": -gene_len,
                                                      "Pol_off":1, "Pol_on":-1})
s.add_reaction("d0 * mRNA", {"mRNA": -1})
s.add_reaction("d1 * Incl", {"Incl": -1})
s.add_reaction("d2 * Skip", {"Skip": -1})
s.add_reaction("s3 * mRNA", {"mRNA": -1, "ret": 1})
s.add_reaction("d3 * ret", {"ret": -1})

#s.simulate_ODE = True
s.simulate()
#s.plot_course()
print(s.get_psi_mean())
#s.plot_course(products=["Skip", "Incl", "Pol_on", "SS_ON", "mRNA", "Pol_pos"], res = ["stoch"])
#s.plot_course(products=["Skip", "Incl", "SS_ON", "mRNA"], res = ["stoch"])
#s.plot_course(products=["U1", "U2_1", "U2_2", "Pol_off", "Incl_nasc", "Skip_nasc"], res = ["stoch"])
psi_means = []
v_elong = np.linspace(1,50,40)



for elong_v in v_elong:
    s.set_param("elong_v", elong_v)
    s.simulate()
    psi_means.append( s.get_psi_mean())
ax = plt.subplot()
ax.plot(v_elong, psi_means, ".", markersize=20)

ax.set_xlabel("Elongation rate")
ax.set_ylabel("Psi")
ax.set_title("Cotranscriptional splicing")