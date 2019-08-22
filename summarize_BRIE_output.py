#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:21:14 2019

@author: timur
"""

#import pyopencl as ocl

import numba as nb
import pandas as pd
import numpy as np
import glob
import os
import pylab as plt 
from sklearn.linear_model import LinearRegression
from bioch_sim import *
import bioch_sim as bs
import aux_th
import scipy as sp 
import random as rd
import matplotlib.cm as cm
from sklearn.decomposition import PCA

from support_th import *

BRIE_dir = os.path.join("/home","timur","ext","working_dir","PRJEB15062", "BRIE_output")
#BRIE_dir = os.path.join("E:\\Eigen Dateien\\Arbeit_IMB\\SysBioSimulation\\dates\\BRIE_output", "BRIE_output")

def _main():

    cell_paths = glob.glob(os.path.join(BRIE_dir, "*"))
    cell_names = [os.path.basename(c_n) for c_n in cell_paths]
    
    mux = pd.MultiIndex.from_product([cell_names, ["counts", "FPKM",  "PSI"]], names = ["cell_ID", "value"])
    
    summary_df = pd.DataFrame(columns=mux)
    
    cell_df = pd.DataFrame()
    for c_path in cell_paths:
        c_name = os.path.basename(c_path)
    #    print("read cell values for: "  + c_name)
        f_c_path = os.path.join(c_path, "fractions.tsv")
        cell_df_raw = pd.read_csv(f_c_path, "\t", low_memory = False)
        indexes = cell_df_raw.tran_id.str.contains("in$").values
        cell_df = cell_df_raw[indexes]
        cell_df_out = cell_df_raw[indexes == False]
        cell_df.counts = cell_df["counts"].values + cell_df_out.counts.values
        cell_df.FPKM = cell_df.FPKM.values + cell_df_out.FPKM.values
        
        summary_df[(c_name, "counts")] = cell_df["counts"]
        summary_df[(c_name, "FPKM")] = cell_df["FPKM"]
        summary_df[(c_name, "PSI")] = cell_df["Psi"]
        
    summary_df.set_index(cell_df["gene_id"].values, inplace=True)
#    df = perform_QC(summary_df, min_counts = 2e5, min_se = 3000, max_share=0.9,
#                    top_se_count=100, min_reads=5, min_cells=15)
    df_f_s = perform_QC(summary_df, min_counts = 2e5, min_se = 2000, max_share=0.8,
                    top_se_count=100, min_reads=5, min_cells=40)
    
#    general_analysis(df, th_suppoints = 20)
#    show_splicing_data(df)
#    _tmp_plot_psi_to_intens(df)
    
    
    return df_f_s
    






if __name__ == "__main__":
    if True:
        df = _main()
        extend_data(df)
#        tmp_plot_psi_to_intens(df)
        show_counts_to_variance(df, log=True)
        mean_counts = df["mean_counts"].values
        
        
