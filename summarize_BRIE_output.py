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

from sklearn.decomposition import PCA

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
    df = perform_QC(summary_df, min_counts = 2e5, min_se = 3000, max_share=0.9,
                    top_se_count=100, min_reads=5, min_cells=15)
    df_f_s = perform_QC(summary_df, min_counts = 2e5, min_se = 4000, max_share=0.8,
                    top_se_count=100, min_reads=5, min_cells=20)
    
#    general_analysis(df, th_suppoints = 20)
#    show_splicing_data(df)
#    _tmp_plot_psi_to_intens(df)
    
    
    return df_f_s

def perform_QC(df= None, min_counts = 1e5, min_se = 3000, max_share = 0.9,
               top_se_count = 100, min_reads = 5, min_cells = 10 ):
    #https://genomebiology.biomedcentral.com/track/pdf/10.1186/s13059-019-1644-0
    # section "gene expression quantification"
    
    # (i) at least 50.000 counts from endogenous genes
    min_counts = min_counts
    
    # (ii) at least 5000 genes with non-zero expr
    min_se = min_se
    
    # (iii) less than 90% of counts are assigned to the top 100 expressed exons per cell
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
    print(df)
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
    df["slope_psi_to_intens"] = np.array(slopes)                
    ax.hist(slopes, bins=100)
    ax.set_title("Slope of LinReg btw. PSI and log(1+counts)")
    ax.set_ylabel("#")
    ax.set_xlabel("slope")
    return (intensity, df_corr, slopes)

def _tmp_show_best_slopes(df=None):
    if df is None:
        return None
    
    first_n = 4
    
    df_sorted = df.sort_values("slope_psi_to_intens", ascending = True)
    fit = plt.figure()
    for  i in range(first_n):
        ax= plt.subplot(first_n, 3, i*3+ 1)
        row = df_sorted.iloc[i]
        psis = row[(slice(None), "PSI")].values
        counts = row[(slice(None), "counts")].values
        ax.plot(psis, counts, ".")
        ax.set_ylabel("counts")
        ax.set_xlabel("pis")
        ax.set_title("psi to count")
        ax= plt.subplot(first_n, 3, i*3+ 2)
        bs.plot_hist(psis,ax = ax)
        ax.set_xlabel("PSI")
        ax= plt.subplot(first_n, 3, i*3+ 3)
        
        
        

def tmp_plot_psi_to_intens(df = None, log = True, th_suppoints = 10 ):
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
def show_PCA(df):
    pca = PCA(n_components=2)
    psis = df.loc[:,(slice(None), "counts")].values.T
    principalComponents = pca.fit_transform(psis)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2'])
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(principalDf["pc1"], principalDf["pc2"])
    return principalDf, pca
    pca.components_
    
def show_splicing_data(df=None, ax=None, best_n = 20, min_psi_th = 0.3):
    
    if ax == None:
        fig, ax = plt.subplots(1,3, squeeze= True)
        
    #calculate Bimodality:
    extend_data(df)

    sorted_df = df.sort_values("Bscore", ascending = False)
    
    if(best_n == None):
        best_n = len(np.argwhere(df["Bscore"] >= min_psi_th))
    
    ax[0].plot(df["mean"], df["std"], ".", c = "blue", label="All PSI values")
    ax[0].set_title("SD over PSI")
    ax[0].set_xlabel("mean(PSI)")
    ax[0].set_ylabel("SD(PSI)")
    means = sorted_df["mean"].values[0:best_n]
    stds = sorted_df["std"].values[0:best_n]
    names  = sorted_df.index.values[0:best_n]
    ax[0].plot(means, stds, ".", c = "red", label="High Bscore")
    ax[0].legend()
    
    ax[1].hist(df["Bscore"].values)
    ax[1].set_title("Bscore distribution")
    ax[1].set_xlabel("Bscore")
    
    aux_th.plot_hist(sorted_df.loc[:,(slice(None),"PSI")].values[0], ax= ax[2])
    ax[2].set_title("Best Bimodality: %s" % sorted_df.index.values[0])
    ax[2].set_xlabel("PSI")
    ax[2].set_ylabel("cell count")
    
    
    all_val = sorted_df.loc[:,(slice(None), "PSI")].values[0:best_n]
    df_bimodal_as = pd.DataFrame(all_val).transpose()
    #heatmap
#    im = ax[3].imshow(df_bimodal_as.corr())
#    cbar = ax[3].figure.colorbar(im, ax=ax[3])
#    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    
    df_corr = df_bimodal_as.corr()
    im, cbar = aux_th.heatmap(df_corr,row_labels = names,
                 col_labels =names , ax = None)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    val_count = np.empty((best_n, best_n))
    #calculate common points
    for k in range(best_n):
        for j in range(best_n):
            val_count[k,j] = (~np.isnan(all_val[k]) & ~np.isnan(all_val[j])).sum() 
    aux_th.annotate_heatmap(im, val_count, valfmt="{x:.0f}", fontsize = 10 )
        
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
#    c_inds = np.argsort(-np.abs(np.array(corr_values)) * np.log(val_c))
    c_inds = np.argsort(-np.abs(np.array(corr_values)))
    
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
        print(fpkms, x, y)
        ax.hist([fpkms[x][~np.isnan(fpkms[x])], fpkms[y][~np.isnan(fpkms[y])]], label = [names[x], names[y]])
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

def extend_data(df):
    if not df.columns.contains("Bscore"):
        df["Bscore"] = [bs.get_bimodal_score(a, tendency=True) for a in df.loc[:,(slice(None),"PSI")].values ]
    if not df.columns.contains("mean"):
        df["mean"] = [np.nanmean(a) for a in df.loc[:,(slice(None),"PSI")].values ]
    if not df.columns.contains("std"):
        df["std"] = [np.nanstd(a) for a in df.loc[:,(slice(None),"PSI")].values ]

def show_counts_to_variance(df = None):
    psi_stds = df["std"]
    psi_means = df["mean"]
    counts = np.nanmean(df.loc[:,(slice(None), "counts")], axis=1)
    counts = np.log(counts)
    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(1,3,1)
    ax.scatter(np.log(counts), psi_stds)
    ax.set_ylabel("std(PSI)")
    ax.set_xlabel("ln(mean(counts))")
    
    ax = fig.add_subplot(1,3,2)
    ax.scatter(psi_means, psi_stds)
    ax.set_ylabel("std(psi)")
    ax.set_xlabel("mean(psi)")
    
    ax = fig.add_subplot(1,3,3)
    psi_std_theory = sp.stats.binom.std(counts, psi_means)/counts
    ax.scatter(psi_stds, psi_std_theory, label="binomial sim")
    ax.set_ylabel("PSI simulated")
    ax.set_xlabel("PSI measured")
    
    psi_std_theory = tmp_simulate_std(counts, psi_means)
    ax.scatter(psi_stds, psi_std_theory, label="gillespie sim")
    ax.legend()
    return fig

sims_tmp = []
psis_tmp =[]
counts_tmp = []
def tmp_simulate_std_gillespie(counts, psi_means):
    global sims_tmp, psis_tmp, counts_tmp
    sims_tmp = [] 
    psis_tmp = []
    counts_tmp = []
    s =bs.get_exmpl_sim("basic")
    s.simulate_ODE = False
    s.set_raster_count(10001)
    s.set_runtime(1000)
    res = np.zeros(len(counts))
    for i in range(len(counts)):
        c = counts[i]
        psi = psi_means[i]
        print("counts: ", c, "\n", "psi_mean: ", psi )
        s1 = s.params["s1"]
        d1 = s.params["d1"]
        s.set_param("d2", d1)
        d2 = s.params["d2"]
        s3 = s.params["s3"]
        d0 = s.params["d0"]
        
        s2 = s1/psi -s1
        pre_ss = c/(s1/d1 + s2/d2)
        v_syn = pre_ss*(s1 + s2 + s3 + d0)
        
        s.set_param("s2", s2)
        s.set_param("v_syn", v_syn)
        
        s.simulate()
        sims_tmp.append(s)
        (indx, psis) = s.compute_psi(ignore_extremes=True)
        res[i] = np.nanstd(psis, ddof=0)
        psis_tmp.append(np.mean(psis))
        counts_tmp.append(np.mean(s.get_res_col("Incl")[100:] + s.get_res_col("Skip")[100:]))
        
    return res


def tmp_simulate_std_binomial(counts, psi_means):
    
    sim_n = 1000
    res_all = np.zeros(len(counts))
    for i in range(len(counts)):
        tries = np.zeros(sim_n)
        for j in range(sim_n):
            res_j = np.random.uniform(size=int(counts[i]))
            res_j = np.where(res_j < psi_means[i], 1, 0)
            tries[j] = np.sum(res_j)/counts[i]
        res_all[i] = np.std(tries, ddof=0)
    return res_all
        

def tmp_compare_binomial_gillespie(counts, psi_means):
    fig = plt.figure()
    binom = sp.stats.binom.std(counts, psi_means)/counts
    gillespie = tmp_simulate_std_gillespie(counts, psi_means)
#    gillespie = tmp_simulate_std_binomial(counts, psi_means)
    ax = fig.add_subplot(111)
    cm = plt.cm.get_cmap('RdYlBu')
    ax.scatter(binom, gillespie, s = psi_means*100, c = psi_means, cmap = cm)
    ax.set_xlabel("Binomial, std")
    ax.set_ylabel("Gillespie, std")
    
    ax.plot([0,np.max(binom)],[0,np.max(binom)], c="r")
    return 0

anz = 1000
psi_means = np.random.uniform(size= anz)
counts = np.random.uniform(5,100, anz)
tmp_compare_binomial_gillespie(counts,psi_means)


if __name__ == "__main__":
    if True:
        df = _main()
        
