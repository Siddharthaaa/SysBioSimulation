#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:21:14 2019

@author: timur
"""

import numba as nb
import pandas as pd
import numpy as np
import glob
import os
import pylab as plt 
from sklearn.linear_model import LinearRegression
from bioch_sim import *
import bioch_sim as bs
from aux_th import *

BRIE_dir = os.path.join("/home","timur","ext","working_dir","PRJEB15062", "BRIE_output")

def _main():

    cell_paths = glob.glob(os.path.join(BRIE_dir, "*"))
    cell_names = [os.path.basename(c_n) for c_n in cell_paths]
    
    mux = pd.MultiIndex.from_product([cell_names, ["counts", "FPKM",  "PSI"]], names = ["cell_ID", "value"])
    
    summary_df = pd.DataFrame(columns=mux)
    
    
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
    df = perform_QC(summary_df, min_counts = 2e5, min_se = 3000, max_share=0.8,
                    top_se_count=130, min_reads=10, min_cells=15)
    general_analysis(df)
    
    return df

def perform_QC(df= None, min_counts = 1e5, min_se = 3000, max_share = 0.9,
               top_se_count = 100, min_reads = 5, min_cells = 10 ):
    #https://genomebiology.biomedcentral.com/track/pdf/10.1186/s13059-019-1644-0
    # section "gene expression quantification"
    
    # (i) at least 50.000 counts from endogenous genes
    min_counts = min_counts
    
    # (ii) at least 5000 genes with non-zero expr
    min_se = min_se
    
    # (iii) less than 90% of counts are assigned to the top 100 expressed genes per cell
    max_share = max_share
    top_se_count = top_se_count
    min_reads = min_reads
    min_cells = min_cells
    
    # (iv) less than 20% of counts are assigned to ERCC spike-in sequences
    # dont know how to get ERCC seqs
    # probable BRIE ignores such seqs
    
    # (v) a Salmon mapping rate of at least 40%
    # cannot perform
    df = df.copy()
    mux_names = df.columns.names
    
    # (i)
    col_idx = np.sum(df.loc[:, (slice(None), "counts" )].values, 0) < min_counts
    cell_ids = df.columns.levels[0].values[col_idx]
#    df = df.loc[:, (cell_ids, slice(None))]
    df.drop(columns = cell_ids, level=0, inplace = True)
    df.columns = pd.MultiIndex.from_tuples(list(df), names = mux_names)
    # (i)
    
    # (ii)
    col_idx = np.sum(df.loc[:, (slice(None), "counts" )].values > 1e-10, 0) < min_se
    cell_ids = df.columns.levels[0].values[col_idx]
    df.drop(columns = cell_ids, level=0, inplace = True)
    df.columns = pd.MultiIndex.from_tuples(list(df), names = mux_names)
    # (ii)
    
    # (iii)
    cell_ids = df.columns.levels[0].values
    c_ids_to_drop = []
    for cell_id in cell_ids:
        reads = df.loc[:, (cell_id, "counts")].values
        reads = np.sort(reads)
        if(np.sum(reads[-top_se_count:]) / np.sum(reads) > max_share):
            c_ids_to_drop.append(cell_id)
    df.drop(columns = c_ids_to_drop, level=0, inplace = True)
    df.columns = pd.MultiIndex.from_tuples(list(df), names = mux_names)
    # (iii)
    
    # (vi)
    # QC Psi values
    # minimum coverage 5 reads in at least 10 cells
    # set PSI to np.nan if reads < min_reads
    gene_ids = df.index.values
    cell_ids = df.columns.levels[0].values
    for c_id in cell_ids:
        reads = df.loc[:, (c_id, "counts")].values
        psis = df.loc[:, (c_id, "PSI")].values
        psis = np.where(reads < min_reads, np.nan, psis )
        df[(c_id, "PSI")] = psis
        
    row_idx = np.sum(np.isnan(df.loc[:, (slice(None), "PSI" )].values) == False , 1) < min_cells
    g_id_to_drop = gene_ids[row_idx]
    df.drop(g_id_to_drop, inplace = True)
    # (vi)
    
    return df

def general_analysis(df = None, ax = None, th_suppoints = 10):
    
    fig = plt.figure()
    ax = plt.subplot(1,3,1)
    stds = np.nanstd(df.loc[:,(slice(None),"PSI")].values,1)
    means = np.nanmean(df.loc[:,(slice(None),"PSI")].values,1)
    intensity = np.nanmean(df.loc[:, (slice(None), "counts")], 1)
#    intensity = np.array([np.array(a, dtype=np.float64) for a in df.loc[:,"counts"].values])
#    intensity = np.array([np.mean(a[np.logical_not(np.isnan(a))]) for a in intensity])
    intensity = np.log(1+ intensity)
#    intensity = intensity / np.max(intensity[np.logical_not(np.isnan(intensity))])
#    colors = cm.rainbow(intensity)
#    print(colors)
    cm = plt.cm.get_cmap('RdYlBu')
    paths = ax.scatter(means, stds, c=intensity, cmap = cm)
    cbar = fig.colorbar(paths, ax = ax)
#    cbar = ax.figure.colorbar(None, ax=ax)
    cbar.ax.set_ylabel("log (mean(counts) + 1)", rotation=-90, va="bottom")
    ax.set_xlabel("mean(PSI)")
    ax.set_ylabel("SD(PSI)")
    
    
    ax = plt.subplot(1,3,2)
    
    all_val = df.loc[:, (slice(None), "PSI")].values
    df_bimodal_as = pd.DataFrame(all_val).transpose()
    
    df_corr = df_bimodal_as.corr()
    
    
    corr_cs = []
    for i in range(0, df_corr.shape[0]):
        for j in range(i+1, df_corr.shape[1]):
            t_inds = np.where(~np.isnan(all_val[i]) & ~np.isnan(all_val[j]))
            if(len(t_inds[0]) >= th_suppoints):
                corr_cs.append(df_corr.iloc[i,j])
            
    ax.hist(corr_cs)
    ax.set_ylabel("#")
    ax.set_xlabel("Correlation R")
    ax.set_title("R dist (PSI-PSI), min points: %d " % th_suppoints)
    
    
    ax = plt.subplot(1,3,3)
    intens = df.loc[:, (slice(None), "counts")].values
    slopes = []
    for i in range(len(all_val)):
        if(intens[i] is not None):
            t_inds = np.where(~np.isnan(all_val[i]) & ~np.isnan(intens[i]))
#            print(t_inds)
            if(len(t_inds[0]) >= th_suppoints):
                x = np.array(all_val[i][t_inds])
                y = np.log(1+intens[i][t_inds])
                model = LinearRegression()
        #            print(x.reshape((-1, 1)))
                model.fit(x.reshape((-1, 1)), y)
                slopes.append(model.coef_[0])
                
    ax.hist(slopes, bins=100)
    ax.set_title("Slope of LinReg btw. PSI and log(1+counts)")
    ax.set_ylabel("#")
    ax.set_xlabel("slope")
    return (intensity, df_corr, slopes)

def _tmp_plot_psi_to_intens(df = None, log = True, th_suppoints = 10 ):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    stds = np.nanstd(df.loc[:,(slice(None),"PSI")].values,1)
    means = np.nanmean(df.loc[:,(slice(None),"PSI")].values,1)
    intensity = np.nanmean(df.loc[:, (slice(None), "counts")], 1)
    if log:
        intensity = np.log(1+ intensity)
#    intensity = intensity / np.max(intensity[np.logical_not(np.isnan(intensity))])
#    colors = cm.rainbow(intensity)
#    print(colors)
    paths = ax.scatter(stds, intensity )
#    cbar = fig.colorbar(paths, ax = ax)
#    cbar = ax.figure.colorbar(None, ax=ax)
#    cbar.ax.set_ylabel("log (mean(counts) + 1)", rotation=-90, va="bottom")
    ax.set_xlabel("std(PSI)")
    ax.set_ylabel("log (mean(counts) + 1)")
    
    x = stds
    y = intensity
    
    print(np.logical_not(np.isnan(x)).sum())
    print(np.logical_not(np.isnan(y)).sum())   
    
    inds = np.where(np.logical_not(np.isnan(x)) * np.logical_not(np.isnan(y)))
    
    x = x[inds]
    y = y[inds]
    
    
    model = LinearRegression()
#            print(x.reshape((-1, 1)))
    model.fit(x.reshape((-1, 1)), y)
    ax.plot(x, model.predict(x.reshape((-1,1))), c= "orange", label = "R2 = %.3f" % model.score(x.reshape((-1,1)), y))
#    ax.legend(fontsize = 10)
  
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.legend()
    
    return  ax

def show_splicing_data(df=None, ax=None, best_n = 20, min_psi_th = 0.3):
    
    
    ness_cols = ["ID", "Bscore", "mean", "std", "PSI_values"]
    
    for ness_col in ness_cols:
        if ness_col not in df:
            raise ValueError("DataFrame must contain following columns:", ness_col)
    if ax == None:
        fig, ax = plt.subplots(1,3, squeeze= True)
        
    #calculate Bimodality:
    if not df.columns.contains("Bscore"):
        df["Bscore"] = [bs.get_bimodal_score(a, tendency=True) for a in df.loc[:,(slice(None),"PSI")].values ]
    if not df.columns.contains("mean"):
        df["mean"] = [np.nanmean(a) for a in df.loc[:,(slice(None),"PSI")].values ]
    if not df.columns.contains("std"):
        df["std"] = [np.nanstd(a) for a in df.loc[:,(slice(None),"PSI")].values ]
#    df.set_index("ID", inplace = True)
    sorted_df = df.sort_values("Bscore", ascending = False)
    
    if(best_n == None):
        best_n = len(np.argwhere(df["Bscore"] >= min_psi_th))
    
    ax[0].plot(df["mean"], df["std"], ".", c = "blue", label="All PSI values")
    ax[0].set_title("SD over PSI")
    ax[0].set_xlabel("mean(PSI)")
    ax[0].set_ylabel("SD(PSI)")
    means = sorted_df["mean"].values
    stds = sorted_df["std"].values
    names  = sorted_df.index.values
    ax[0].plot(means, stds, ".", c = "red", label="High Bscore")
    ax[0].legend()
    
    ax[1].hist(df["Bscore"])
    ax[1].set_title("Bscore distribution")
    ax[1].set_xlabel("Bscore")
    
    plot_hist(sorted_df.loc[:,(slice(None),"PSI")].values[0], ax= ax[2])
    ax[2].set_title("Best Bimodality: %s" % sorted_df.index.values[0])
    ax[2].set_xlabel("PSI")
    ax[2].set_ylabel("cell count")
    
    
    all_val = sorted_df.loc[:,(slice(None), "PSI")].values[0:best_n]
    df_bimodal_as = pd.DataFrame(all_val).transpose()
    #heatmap
#    im = ax[3].imshow(df_bimodal_as.corr())
#    cbar = ax[3].figure.colorbar(im, ax=ax[3])
#    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    from bioch_sim import *
    
    df_corr = df_bimodal_as.corr()
    im, cbar = heatmap(df_corr,row_labels = names,
                 col_labels =names , ax = None)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    val_count = np.empty((best_n, best_n))
    #calculate common points
    for k in range(best_n):
        for j in range(best_n):
            val_count[k,j] = (~np.isnan(all_val[k]) & ~np.isnan(all_val[j])).sum() 
    annotate_heatmap(im, val_count, valfmt="{x:.0f}", fontsize = 10 )
        
    ncols = int(np.sqrt(best_n)+1)
    nrows = int(best_n/ncols+1)    
    intens = sorted_df.loc[:,(slice(None), "counts")].values[0:best_n]
    fig = plt.figure()
    for i in range(best_n):
        ax = plt.subplot(nrows, ncols, i+1)
        t_inds = np.where(~np.isnan(all_val[i]) & ~np.isnan(intens[i]))
        x = np.array(all_val[i][t_inds])
        y = intens[i][t_inds]
        ax.plot(all_val[i], intens[i],".")
        model = LinearRegression()
#            print(x.reshape((-1, 1)))
        model.fit(x.reshape((-1, 1)), y)
        ax.plot(x, model.predict(x.reshape((-1,1))), label = "R2 = %.3f" % model.score(x.reshape((-1,1)), y))
        ax.legend(fontsize = 10)
        ax.set_title(names[i], fontsize = 10)
        ax.set_xlabel("PSI", fontsize =10)
        ax.set_ylabel("counts", fontsize =10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
            
    fpkms = sorted_df.loc[:,(slice(None),"FPKM")].values[0:best_n]
    counts = intens
    fig = plt.figure()
    for i in range(best_n):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.plot(counts[i], fpkms[i],".")
        ax.set_title(names[i])
        ax.set_xlabel("counts")
        ax.set_ylabel("fpkms")    
    
    #search for the best correlation
    corr_values =[]
    corr_indices = []
    val_c = []
    for i in range(df_corr.shape[0]):
        for j in range(i+1, df_corr.shape[1]):
            corr_values.append(df_corr.iloc[i,j])
            val_c.append(val_count[i,j])
            corr_indices.append([i,j])
    
    #best correlated exons
    #with corrected R
    c_inds = np.argsort(-np.abs(np.array(corr_values)) * np.log(val_c))
    
#    print(c_inds)
#    print(np.abs(np.array(corr_values)[c_inds]))
#    print(np.array(val_c)[c_inds])
    
    first_n = 4
    fig = plt.figure()
    j = 1
    for i in c_inds[0:first_n]:
        ax = plt.subplot(first_n, 3, j)
        
        x = corr_indices[i][0]
        y = corr_indices[i][1]
        
        ax.plot(all_val[x], all_val[y], ".")
        ax.set_ylabel(names[y], fontsize = 8)
        ax.set_xlabel(names[x], fontsize = 8)
        ax.set_title("R = %.3f" % df_corr.iloc[x,y])
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        ax = plt.subplot(first_n,3 , j+1)
        ax.hist([fpkms[x], fpkms[y]], label = [names[x], names[y]])
        ax.legend(fontsize = 10)
        ax.set_xlabel("fpkm")
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        
        ax = plt.subplot(first_n, 3, j+2)
        ax.plot(all_val[x], fpkms[x], ".", label = names[x] )
        ax.plot(all_val[y], fpkms[y], ".", label = names[y] )
        ax.set_ylabel("fpkm", fontsize = 10)
        ax.set_xlabel("PSI", fontsize = 10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.legend(fontsize = 10)
        ax.set_title("")
        j+=3
    
    
    indx_pairs = np.argsort(df_corr)
    return (sorted_df, indx_pairs, df_corr)

if __name__ == "__main__":
    if True:
        df = _main()
