# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 07:29:06 2019

@author: Timur

read data from Fig 2 of 
https://www.embopress.org/doi/10.15252/msb.20156278#msb156278-fig-0002

"""

import pandas as pd
import numpy as np
from summarize_BRIE_output import *

excl_path = "E:\\Eigen Dateien\\Arbeit_IMB\\SysBioSimulation\\dates\\msbfig2.xlsx"

if __name__ == "__main__":
    counts_df = pd.read_excel(excl_path, 1)   
    psi_df = pd.read_excel(excl_path, 2)
    cell_types = counts_df["cell.type"].values 
    cell_types = np.unique(cell_types)
        
    res_dfs = []
    
    cell_ids = counts_df.filter(regex="^\d+$")
    
    for cell_t in cell_types:
#        res_df = pd.DataFrame()
        mux = pd.MultiIndex.from_product([cell_ids, ["counts", "FPKM",  "PSI"]], names = ["cell_ID", "value"])
        res_df = pd.DataFrame(columns=mux)
        for df , v_name in zip ([psi_df, counts_df, counts_df], ["PSI", "counts", "FPKM"]):
            
            filtered_df = df[counts_df["cell.type"] == cell_t]
            
            genes = filtered_df["gene.name"].values
            
            c_df = filtered_df.filter(regex="^\d+$")
            
            for col in c_df:
                res_df[(col, v_name)] = c_df[col].values
                res_df.set_index(genes, inplace=True)              
        res_dfs.append(res_df)
    d = res_dfs[0]
