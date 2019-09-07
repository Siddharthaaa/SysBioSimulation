# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:35:50 2019

@author: timuhorn

testing estimated pars 
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


df = read_data()

counts = df["mean_counts"].values
psis = df["mean"].values
psi_stds = df["std"].values


class VectorDistance(pa.Distance):
    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        
        l_sum = 0
        for k in x.keys():
            l = np.linalg.norm(x[k] - x_0[k])
            l_sum += l
        return l_sum
    
distance = VectorDistance()

pars1 = np.linspace(0.1,1,6)
for par1 in pars1:
    dists = []
    pars2 = np.linspace(0, 1, 20)             
    for r in pars2:
        stds = sth.tmp_simulate_std_gillespie(counts, psis,
                                          sim_rnaseq=r,
                                          extrapolate_counts=par1)
        d = distance({"x": psi_stds}, {"x": stds})
        dists.append(d)
    
    plt.plot(pars2, dists, label = "extrapol: %1.2f" % par1)

plt.legend()

def read_data():
    excl_path = "dates/msbfig3.xlsx"
    
    
    counts_df = pd.read_excel(excl_path, 1)   
    psi_df = pd.read_excel(excl_path, 2)
    cell_types = counts_df["cell.population"].values 
    cell_types = np.unique(cell_types)
        
    res_dfs = []
    
    cell_ids = counts_df.filter(regex="^\d+$")
    
    for cell_t in cell_types:
    #        res_df = pd.DataFrame()
        mux = pd.MultiIndex.from_product([cell_ids, ["counts", "FPKM",  "PSI"]], names = ["cell_ID", "value"])
        res_df = pd.DataFrame(columns=mux)
        for df , v_name in zip ([psi_df, counts_df, counts_df], ["PSI", "counts", "FPKM"]):
            
            filtered_df = df[counts_df["cell.population"] == cell_t]
            
            genes = filtered_df["gene.name"].values
            
            c_df = filtered_df.filter(regex="^\d+$")
            
            for col in c_df:
                res_df[(col, v_name)] = c_df[col].values if v_name != "counts" else np.exp(c_df[col].values)
    #                res_df[(col, v_name)] = c_df[col].values 
                res_df.set_index(genes, inplace=True)
    #        perform_QC(res_df, min_counts=100,min_se= 10, max_share=1, top_se_count=1, min_reads=5, min_cells=10 )
        res_dfs.append(res_df)
    
    d = res_dfs[1]
    d = sth.perform_QC(d, min_counts=100,min_se= 10, max_share=1, top_se_count=1, min_reads=5, min_cells=10 )
    sth.extend_data(d)
    return d
