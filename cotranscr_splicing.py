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

gene_len = 3000
u1_1_bs_pos = 150
u2_1_bs_pos = 1500
u1_2_bs_pos = 1700
u2_2_bs_pos = 2800


params = {"pr_on": 2, "pr_off" : 0.1,
        "elong_v": 1000,
        "gene_len": gene_len,
        "spl_rate": 1,
        "u1_1_bs_pos": u1_1_bs_pos , # U1 binding site position
        "u1_2_bs_pos": u1_2_bs_pos ,
        "u1_1_br": 1,  # binding rate of U1
        "u1_1_ur": 0.2, # unbinding rate of U1
        "u1_2_br": 1,  # binding rate of U1
        "u1_2_ur": 0.2, # unbinding rate of U1
        "u2_1_bs_pos": u2_1_bs_pos, # U2 bind. site pos 1
        "u2_2_bs_pos": u2_2_bs_pos,
        "u2_1_br": 1,
        "u2_2_br": 1,
        "u2_1_ur": 0.1,
        "u2_2_ur": 0.11,
        "tr_term_rate": 1000,
        "s1":1, "s2":1, "s3": 0.1,
        "d0":0.01, "d1": 0.01, "d2":0.01, "d3":0.1
        }


s = bs.SimParam("Cotranscriptional splicing", 10000, 100001, params = params,
                init_state = {"Pol_on":0, "Pol_off": 1,
                              "nascRNA_bc": 0,
                              "Pol_pos": 0,
                              "Skip":0, "Incl": 0, "ret": 0,
                              "U1_1":0, "U1_2":0,   #Splicosome units U1 binded
                              "U2_1":0, "U2_2":0,
                              "Intr1":0, "Intr2":0, "Exon1":0,
                              "Intr1_ex":0, "Intr2_ex":0, "Exon1_ex":0})

s.add_reaction("pr_on * Pol_off", 
               {"Pol_on":1, "Pol_off": -1,"Exon1":1, "Intr1":1,"Intr2":1, "Tr_dist": gene_len},
               name = "Transc. initiation")

s.add_reaction("elong_v * Pol_on if Pol_pos < gene_len else 0",
               {"nascRNA_bc": 0, "Pol_pos":1, "Tr_dist":-1},
               name = "Elongation")

# Ux un/binding cinetics
s.add_reaction("u1_1_br * Intr1 if Pol_pos > u1_1_bs_pos and U1_1 < 1 else 0",
               {"U1_1":1}, "U1_1 binding")
s.add_reaction("u1_1_ur * U1_1", {"U1_1":-1}, "U1_1 unbinding")

s.add_reaction("u1_2_br * Intr2 if Pol_pos > u1_2_bs_pos and U1_2 < 1 else 0",
               {"U1_2":1}, "U1_2 binding")
s.add_reaction("u1_2_ur * U1_2", {"U1_2":-1}, "U1_2 unbinding")

s.add_reaction("u2_1_br * Intr1 if Pol_pos > u2_1_bs_pos and U2_1 < 1 else 0",
               {"U2_1":1}, "U2_1 binding")
s.add_reaction("u2_1_ur * U2_1", {"U2_1":-1}, "U2_1 unbinding")

s.add_reaction("u2_2_br * Intr2 if Pol_pos > u2_2_bs_pos and U2_2 < 1 else 0",
               {"U2_2":1}, "U2_2 binding")
s.add_reaction("u2_2_ur * U2_2", {"U2_2":-1}, "U2_2 unbinding")

#Splicing
s.add_reaction("U1_1 * U2_1 * Intr1 * spl_rate",
               {"Intr1":-1, "U1_1":-1, "U2_1":-1, "Intr1_ex":1},
               name="Intron 1 excision")
s.add_reaction("U1_2 * U2_2 * Intr2 * spl_rate",
               {"Intr2":-1, "U1_2":-1, "U2_2":-1, "Intr2_ex":1},
               name="Intron 2 excision")
s.add_reaction("U1_1 * U2_2 * Intr1 * Intr2 * Exon1 * spl_rate",
               {"Intr1":-1, "Intr2":-1, "Exon1":-1, "U1_1":-1, "U2_2":-1, "Exon1_ex":1, "Intr1_ex":1, "Intr2_ex":1},
               name="Exon 1 excision (inclusion)")

#Transcription termination
s.add_reaction("tr_term_rate * Exon1_ex * Intr1_ex * Intr2_ex if Tr_dist == 0 else 0",
               {"Exon1_ex":-1, "Intr1_ex":-1, "Intr2_ex":-1, "Skip":1,
                "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
               name = "Termination: skipping")
s.add_reaction("tr_term_rate * Exon1 * Intr1_ex * Intr2_ex if Tr_dist == 0 else 0",
               {"Exon1":-1, "Intr1_ex":-1, "Intr2_ex":-1, "Incl":1,
                "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
               name = "Termination: inclusion")
s.add_reaction("tr_term_rate * Exon1 * Intr1_ex * Intr2 if Tr_dist == 0 else 0",
               {"Exon1":-1, "Intr1_ex":-1, "Intr2":-1, "ret_i2":1,
                "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
               name = "Termination: Intron 1 retention")
s.add_reaction("tr_term_rate * Exon1 * Intr1 * Intr2_ex if Tr_dist == 0 else 0",
               {"Exon1":-1, "Intr1":-1, "Intr2_ex":-1, "ret_i1":1,
                "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
               name = "Termination: Intron 2 retention")
s.add_reaction("tr_term_rate * Exon1 * Intr1 * Intr2 if Tr_dist == 0 else 0",
               {"Exon1":-1, "Intr1":-1, "Intr2":-1, "ret":1,
                "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
               name = "Termination: full retention")

s.add_reaction("d0 * mRNA", {"mRNA": -1})
s.add_reaction("s1 * mRNA", {"mRNA": -1, "Incl": 1})
s.add_reaction("s2 * mRNA", {"mRNA": -1, "Skip": 1})
s.add_reaction("d1 * Incl", {"Incl": -1})
s.add_reaction("d2 * Skip", {"Skip": -1})
s.add_reaction("s3 * mRNA", {"mRNA": -1, "ret": 1})
s.add_reaction("d3 * ret", {"ret": -1})
s.add_reaction("d3 * ret_i1", {"ret_i1": -1})
s.add_reaction("d3 * ret_i2", {"ret_i2": -1})


print(s.compile_system())
#s.simulate_ODE = True
s.simulate(max_steps=1e8)
#s.plot_course()
res = s.results["stoch_rastr"]

print(s.get_psi_mean())
#s.plot_course(products=["Skip","Incl", "Pol_on","Pol_off", "mRNA", ], res = ["stoch"])
#s.plot_course(products=["Skip","Incl", "U1_1", "Intr1", "Intr2", "Exon1"], res = ["stoch"])
#s.plot_course(products=["Pol_pos","Tr_dist"], res = ["stoch"])
#s.plot_course(products=["Skip", "Incl", "Pol_pos", "mRNA"], res = ["stoch"])
#s.plot_course(products=["U1", "U2_1", "U2_2", "Pol_off", "Incl_nasc", "Skip_nasc"], res = ["stoch"])

psi_means = []

v_elong = np.linspace(100,1500,30)

for elong_v in v_elong:
    s.set_param("elong_v", elong_v)
    s.simulate()
    psi_means.append( s.get_psi_mean())
ax = plt.subplot()
ax.plot(v_elong, psi_means, ".", markersize=20)

ax.set_xlabel("Elongation rate")
ax.set_ylabel("Psi")
ax.set_title("Cotranscriptional splicing")