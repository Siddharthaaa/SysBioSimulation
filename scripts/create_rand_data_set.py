# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:59:52 2019
Discription: create(simulate) a random dataset 
@author: timuhorn
"""

from bioch_sim import *
from aux_th import *
import numpy as np
import pandas as pd
#import pyabc




v_syn = 10
s1= 4
s2 = 2
s3 = 0.1

incl_ss = 10
skip_ss = 5
ret_ss = 3
d0 = 0.1
d1 = d2= 1
d3 = 1

pre_ss = v_syn/(s1+s2+s3+d0)
incl_ss=pre_ss*s1/d1
skip_ss=pre_ss*s2/d2
ret_ss=pre_ss*s3/d3

psi = incl_ss/(incl_ss + skip_ss)

name="v_syn=%2.2f, s1=%2.2f (psi:%1.2f, I:%2.2f, S:%2.2f)" % (v_syn, s1, psi, incl_ss, skip_ss)
s = SimParam(name,100, 200,
             params = {"v_syn": v_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
             init_state = {"pre_RNA": int(pre_ss), "Incl": int(incl_ss),
                              "Skip": int(skip_ss), "ret": int(ret_ss)})


s.simulate_ODE = False

s.add_reaction("v_syn", {"pre_RNA":1} )
s.add_reaction("d0*pre_RNA", {"pre_RNA":-1} )
s.add_reaction("s1*pre_RNA", {"pre_RNA":-1, "Incl":1})
s.add_reaction("d1*Incl", {"Incl": -1}  )
s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1})
s.add_reaction("d2*Skip", {"Skip":-1}  )
s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1} )
s.add_reaction("d3*ret",  {"ret": -1} )
s.expected_psi = psi

s.compile_system(True)

df = create_DataFrame_template()
for i in range(10):
    s.set_param("s1", np.random.uniform(0,2))
    s.set_param("v_syn", np.abs(np.random.normal(10, 5)))
    
    s.simulate()
    psis = s.compute_psi(ignore_fraction=0)[1]
    mean = np.mean(psis)
    std = np.std(psis)
    b_score = s.get_bimodality()
    counts = np.sum(np.array((s.get_res_col("pre_RNA"), s.get_res_col("Incl"), s.get_res_col("Skip"))), axis = 0)
    gene_id = "GENE_" + str(i)
    
    df.loc[gene_id, "ID"] = gene_id
    df.loc[gene_id, "PSI_values"] = psis
    df.loc[gene_id, "mean"] = mean
    df.loc[gene_id, "std"] = std
#    df["counts"] = df["counts"].astype(object)
    df.loc[gene_id, "counts"] = counts

tmp = general_analysis(df)