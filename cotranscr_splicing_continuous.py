#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:39:26 2019

@author: timur

Modelling cotranscriptional splicing
with

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
v_syn = 10
gene_len = 100
u1_bs_pos = 10
u2_bs_pos_1 = 30
u2_bs_pos_2 = 60


params = {"pr_on": 2, "pr_off" : 0.1,
        "elong_v": 100,
        "gene_len": gene_len,
        "u1_bs_pos": u1_bs_pos , # U1 binding site position
        "u1_br": 1,  # binding rate of U1
        "u1_ur": 0.02, # unbinding rate of U1
        "u2_bs_pos_1": 30, # U2 bind. site pos 1
        "u2_bs_pos_2": 60,
        "u2_br_p1": 1,
        "u2_br_p2": 1,
        "u2_ur_p1": 0.1,
        "u2_ur_p2": 0.11,
        "s1":1, "s2":1, "s3": 0.1,
        "d0":0.01, "d1": 0.01, "d2":0.01, "d3":0.1
        }


s = bs.SimParam("Cotranscriptional splicing", 1000, 10001, params = params,
                init_state = {"Prom_on":0, "Prom_off": 1,
                              "nascRNA_bc": 0,
                              "Pol_pos": 0,
                              "Skip":0, "Incl": 0, "ret": 0,
                              "SS_ON":0})

s.add_reaction("pr_on * Prom_off", {"Prom_on":1, "Prom_off": -1, "SS_ON": 1} )
#s.add_reaction("pr_off * Prom_on", {"Prom_on":-1, "Prom_off": 1} )
s.add_reaction("elong_v * Prom_on if Pol_pos <= gene_len else 0", {"nascRNA_bc": 1, "Pol_pos":1})
s.add_reaction("1000 if Pol_pos >= gene_len else 0", {"mRNA":1, "Pol_pos": -gene_len,
                                                      "nascRNA_bc": -gene_len,
                                                      "Prom_off":1, "Prom_on":-1,
                                                      "SS_ON": 1})
s.add_reaction("u1_br if Pol_pos > u1_bs_pos and U1 < 1  else 0", {"U1":1 } )
s.add_reaction("u1_ur if U1 > 0  else 0", {"U1":-1} )
s.add_reaction("u2_br_p1*SS_ON if Pol_pos > u2_bs_pos_1 and U2_1 < 1  else 0", {"U2_1":1 } )
s.add_reaction("u2_ur_p1 if U2_1 > 0 else 0", {"U2_1":-1 })
s.add_reaction("u2_br_p2*SS_ON if Pol_pos > u2_bs_pos_2 and U2_2 < 1  else 0", {"U2_2":1 } )
s.add_reaction("u2_ur_p2 if U2_2 > 0 else 0", {"U2_2":-1 })
s.add_reaction("U1 * U2_1 * s1", {"Incl":1, "U1": -1, "U2_1": -1, "SS_ON":-1})#,"nascRNA_bc": -(u2_bs_pos_1 - u1_bs_pos) })
s.add_reaction("U1 * U2_2 * s2", {"Skip":1, "U1": -1, "U2_2": -1, "SS_ON":-1})#,  "nascRNA_bc": -(u2_bs_pos_2 - u1_bs_pos) })
s.add_reaction("d0 * mRNA", {"mRNA": -1})
s.add_reaction("d1 * Incl", {"Incl": -1})
s.add_reaction("d2 * Skip", {"Skip": -1})
s.add_reaction("s3 * mRNA", {"mRNA": -1, "ret": 1})
s.add_reaction("d3 * ret", {"ret": -1})

s.simulate_ODE = True
s.simulate()
#s.plot_course()
print(s.get_psi_mean())
s.plot_course(products=["Skip", "Incl", "SS_ON"], res = ["stoch"])

