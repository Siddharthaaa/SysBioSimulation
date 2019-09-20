#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:38:09 2019

@author: timur

Script containing some auxiliary and experimental functions
"""


import numba as nb
import pandas as pd
import numpy as np
import glob
import os
import pylab as plt 
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from bioch_sim import *
import bioch_sim as bs
import aux_th
import scipy as sp 
import random as rd
import matplotlib.cm as cm
from sklearn.decomposition import PCA

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        fig, ax =  plt.subplots()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize = 10)
    ax.set_yticklabels(row_labels, fontsize = 10)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["white", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_3d(x, y, zz):
    fig = plt.figure(figsize=(7,7))
    ax = fig.gca(projection='3d')
    
    X, Y = np.meshgrid(x,y)
    #ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    ax.plot_surface(X,Y,zz.T,  cmap=cm.coolwarm)
    #ax.plot_wireframe(x,y,z,  cmap=cm.coolwarm)
    
    plt.show()
    return ax

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
    col_idx = np.sum(df.loc[:, (slice(None), "counts" )].values > min_reads, 0) < min_se
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
#    print(df)
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
    
#    print(np.logical_not(np.isnan(x)).sum())
#    print(np.logical_not(np.isnan(y)).sum())   
    
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


def extend_data(df, th_psi_std=False):
    if not "Bscore" in df:
        df["Bscore"] = [bs.get_bimodal_score(a, tendency=True) for a in df.loc[:,(slice(None),"PSI")].values ]
    if not "mean" in df:
        df["mean"] = [np.nanmean(a) for a in df.loc[:,(slice(None),"PSI")].values ]
    if not "mean_counts" in df:
        df["mean_counts"] = [np.nanmean(a) for a in df.loc[:,(slice(None),"counts")].values ]
    if not "std" in df:
        df["std"] = [np.nanstd(a) for a in df.loc[:,(slice(None),"PSI")].values ]
    if th_psi_std and not "th_psi_std" in df:
        psi_stds = df["std"].values
        counts_means = df["mean_counts"].values
        df["th_psi_std"] = sp.stats.binom.std(counts_means, psi_stds)/counts_means

def show_counts_to_variance(df = None, gillespie = False, log = False,
                            keep_quantile=0.9, rnaseq_efficiency = 1, extrapolate_counts = None,
                            var_stab_std = False):
    
    res = empty()
    
    
    counts = np.nanmean(df.loc[:,(slice(None), "counts")], axis=1)
    df = df[counts <= np.quantile(counts, keep_quantile)] 
    psi_stds = df["std"].values
    if(var_stab_std):
        psi_stds = np.nanstd(np.arcsin(np.sqrt(df.loc[:, (slice(None), "PSI")].values)), axis = 1)
    psi_means = df["mean"].values
    counts = np.nanmean(df.loc[:,(slice(None), "counts")], axis=1)
    
    if log:
        counts = np.log(counts)
    fig = plt.figure(figsize = (12,6))
    
    psis_tmp = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    cols = cm.rainbow(np.array(psis_tmp)/0.5)
    
#    ax.scatter(counts, psi_stds, label = "")
    ax = fig.add_subplot(111)
    
    psi_stds_theo_gil = tmp_simulate_std_gillespie(counts, psi_means,
                                                       runtime=1000, sim_rnaseq=rnaseq_efficiency,
                                                       extrapolate_counts=extrapolate_counts,
                                                       var_stab = var_stab_std)
    residues = psi_stds_theo_gil - psi_stds
    res.residues = residues
    
    points = 25
    counts_theo = np.linspace(counts.min(), counts.max(), points)
    low_lim = 0
    i=1
    
    for psi, col  in zip (psis_tmp, cols):
        high_lim = 1 - low_lim
        psi_high = 1 - psi
#        ax = fig.add_subplot(2,len(psis_tmp)/2,i)
        ax.set_ylabel("std(PSI)")
        ax.set_xlabel("mean(counts)")
        indx_l = np.where((psi_means >= low_lim) * (psi_means < psi))
        indx_h = np.where((psi_means <= high_lim) * (psi_means > psi_high))
        low_lim = psi
        indx = np.union1d(indx_l, indx_h)
        p_stds = psi_stds[indx]
        cs = counts[indx]
        ax.plot(cs, p_stds,".", c=col, label = "", markersize=10)
        
        ax.plot(cs, psi_stds_theo_gil[indx],"x",c =col, markersize=10)
        for i in range(len(cs)):
            y1, y2 = psi_stds_theo_gil[indx][i], p_stds[i]
            x = cs[i]
            ax.plot([x,x], [y1,y2], "--", c = col)
        
        if(var_stab_std is False):
            psis_th = np.ones(points)*psi
            psi_stds_theo_bin = sp.stats.binom.std(counts_theo, psis_th )/counts_theo
            ax.plot(counts_theo, psi_stds_theo_bin, label = "psi: %2.2f" % psi,c = col, lw=1)
        
#        psi_stds_theo_gil = tmp_simulate_std_gillespie(counts_theo, psis_th,
#                                                       runtime=1000, sim_rnaseq=rnaseq_efficiency,
#                                                       extrapolate_counts=extrapolate_counts)
#        ax.plot(counts_theo, psi_stds_theo_gil,"--",c = col, lw=1)
        
#        ax = fig.add_subplot(1,3,2)
        i+=1
    leg1 = ax.legend()   
    l1 = Line2D([0],[0], marker="x", color = "black", markersize=10, linewidth=0, label = "simulated")
    l2 = Line2D([0],[0],marker="o", color = "black", markersize=10,linewidth=0, label = "measured")
    leg2 = ax.legend([l1,l2],["Simulated", "Measured"], loc = "upper center")
    ax.add_artist(leg1)
    fig = plt.figure(figsize = (16,6))
    ax = fig.add_subplot(111)
    ax.hist(residues, bins = 40)
    ax.axvline(np.mean(residues), c="red")
    ax.set_title("Residues")
    
    
    #3D plot with the same information
#    counts_ln = np.linspace(5,np.quantile(counts, 0.9),50)
    counts_ln = np.linspace(3, np.max(counts),40)
    psi_means_ln = np.linspace(0,1,40)
    
    psi_std_th = np.zeros((len(counts_ln), len(psi_means_ln)))
    for i in range(len(counts_ln)):
        for j in range(len(psi_means_ln)):
            psi_std_th[i,j] = sp.stats.binom.std(counts_ln[i], psi_means_ln[j])/counts_ln[i]
    fig = plt.figure(figsize=(7,7))
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(counts_ln,psi_means_ln)
    #ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    ax.plot_surface(X,Y,psi_std_th.T,  cmap=cm.coolwarm)
    ax.scatter3D(counts, psi_means, psi_stds)
    
    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(1,2,1)
    ax.scatter(psi_means, psi_stds, label = "")
    ax.set_ylabel("std(psi)")
    ax.set_xlabel("mean(psi)")
    #TODO
    for count in np.quantile(counts, [0.05, 0.1, 0.2, 0.3, 0.5, 0.7,1]):
        psis_tmp = np.linspace(0,1, 50)
        stds_tmp = sp.stats.binom.std(count, psis_tmp )/count
        ax.plot(psis_tmp, stds_tmp, label = "counts= %d" % count, lw=0.7)
    ax.legend()
    ax = fig.add_subplot(1,2,2)
    psi_std_theory = sp.stats.binom.std(counts, psi_means)/counts
    ax.scatter(psi_stds, psi_std_theory, label="binomial sim")
    ax.set_ylabel("std(PSI) simulated")
    ax.set_xlabel("std(PSI) measured")
    
    if(gillespie):
        psi_std_theory = tmp_simulate_std_gillespie(counts, psi_means)
        ax.scatter(psi_stds, psi_std_theory, label="gillespie sim")
    ax.plot([0,0.5],[0, 0.5], c = "r")
    ax.legend()
    
#    fig = plt.figure(figsize = (12,6))
#    ax = fig.add_subplot(1,3,1)
#   
#    ax.scatter(counts, psi_stds, label="measured")
    res.fig = fig
    return res

def tmp_simulate_std_gillespie(counts, psi_means, runtime=1000,
                               exact_counts = False,
                               sim_rnaseq=None,
                               extrapolate_counts = None,
                               var_stab = False):
   
    s =bs.get_exmpl_sim("basic")
    s.simulate_ODE = False
    s.set_raster_count(20001)
    s.set_runtime(runtime)
    res = np.zeros(len(counts))
    
    if(extrapolate_counts is not None):
        counts = counts / extrapolate_counts
    
    for i in range(len(counts)):
        c = counts[i]
        psi = psi_means[i]
#        print("counts: ", c, "\n", "psi_mean: ", psi )
        s.set_param("s1", 10)
        s1 = s.params["s1"]
        d1 = s.params["d1"]
        s.set_param("d2", d1)
        d2 = s.params["d2"]
        s3 = s.params["s3"]
        d0 = s.params["d0"]
        
        s2 = (s1/psi -s1) if psi > 0 else 0
        pre_ss = c/(s1/d1 + s2/d2)
        v_syn = pre_ss*(s1 + s2 + s3 + d0)
        
        s.set_param("s2", s2)
        
#        s.set_param("s3", 0)
        s.set_param("v_syn", v_syn)
        
        s.simulate()
        (indx, psis) = s.compute_psi(ignore_extremes=False, recognize_threshold=1,
                                    exact_sum= c if exact_counts else None,
                                    sim_rnaseq=sim_rnaseq, ignore_fraction=0.8)
        if var_stab:
            psis = np.arcsin(np.sqrt(psis))
        res[i] = np.nanstd(psis, ddof=1)
    del s
    return res


def tmp_simulate_std_binomial(counts, psi_means):
    
    sim_n = 1000
    res_all = np.zeros(len(counts))
    for i in range(len(counts)):
        tries = np.zeros(sim_n)
        for j in range(sim_n):
            res_j = np.random.uniform(size=int(counts[i]))
            res_j = np.where(res_j < psi_means[i], 1, 0)
            tries[j] = np.sum(res_j)/int(counts[i])
        res_all[i] = np.std(tries, ddof=1)
    return res_all
        
def simulate_dropout(counts, psi_means, d_out = 0.5):
    res = np.zeros(len(counts))
    i = 0
    for c, psi in zip(counts, psi_means):
        M = c/d_out
        n = M*psi
        N = c
        res[i] = sp.stats.hypergeom.std(M,n,N)/N
        i+=1
    return res

def tmp_compare_binomial_gillespie(counts, psi_means, exact_counts=False):
    fig = plt.figure()
    binom = sp.stats.binom.std(counts, psi_means)/counts
    gillespie = tmp_simulate_std_gillespie(counts, psi_means, runtime=1e5, exact_counts=exact_counts)
#    gillespie = tmp_simulate_std_binomial(counts, psi_means)
#    gillespie = simulate_dropout(counts, psi_means,0.9)
    ax = fig.add_subplot(111)
    cm = plt.cm.get_cmap('RdYlBu')
    paths = ax.scatter(binom, gillespie, s = psi_means*300, c = counts, cmap = cm, alpha=0.35)
    cbar = fig.colorbar(paths, ax = ax)
#    cbar = ax.figure.colorbar(None, ax=ax)
    cbar.ax.set_ylabel("counts", rotation=-90, va="bottom")
    ax.set_xlabel("Binomial, std")
    ax.set_ylabel("Gillespie, std")
    
    ax.plot([0,np.max(binom)],[0,np.max(binom)], c="r")
    return 0

def tmp_compare_gillespie(counts, psi_means, exact_counts=False, sim_rnaseq= 0.5):
    fig = plt.figure()
    gillespie_biased = tmp_simulate_std_gillespie(counts, psi_means, runtime=1e5,
                                                  exact_counts=exact_counts,
                                                  sim_rnaseq = sim_rnaseq)
    gillespie_exact = tmp_simulate_std_gillespie(counts, psi_means, runtime=1e5,
                                                 exact_counts=exact_counts)
#    gillespie = tmp_simulate_std_binomial(counts, psi_means)
#    gillespie = simulate_dropout(counts, psi_means,0.9)
    ax = fig.add_subplot(111)
    cm = plt.cm.get_cmap('RdYlBu')
    paths = ax.scatter(gillespie_exact, gillespie_biased, s = psi_means*300, c = counts, cmap = cm, alpha=0.35)
    cbar = fig.colorbar(paths, ax = ax)
#    cbar = ax.figure.colorbar(None, ax=ax)
    cbar.ax.set_ylabel("counts", rotation=-90, va="bottom")
    ax.set_xlabel("Gillespie, std")
    ax.set_ylabel("Gillespie + rnaSeq sim, std")
    
    ax.plot([0,np.max(gillespie_exact)],[0,np.max(gillespie_exact)], c="r")
    return 0


def filter_assumed_hkg(df = None, psi = (0.2, 0.8), counts_max_cv = 0.2,
                       min_counts_p = 0.1, min_psis_p = 0.1 ):
    df = df.copy()
    df_t = df[(df["mean"].values > psi[0]) * (df["mean"].values < psi[1])]
    
    counts_all = df_t.loc[:, (slice(None), "counts")].values
    psis_all = df_t.loc[:, (slice(None), "PSI")].values
    
    counts_ps = []
    counts_Ds = []
    for counts in counts_all:   
        mean = np.nanmean(counts)
        std = np.nanstd(counts)
        
        (D, p) = sp.stats.kstest(counts, "norm", args = (mean,std) )
        counts_ps.append(p)
        counts_Ds.append(D)
    counts_ps = np.array(counts_ps)
    counts_Ds = np.array(counts_Ds)
    
    psis_ps = []
    psis_Ds = []
    for psis in psis_all:   
        psis = psis[~np.isnan(psis)]
        mean = np.nanmean(psis)
        std = np.nanstd(psis)
        
        (D, p) = sp.stats.kstest(psis, "norm", args = (mean,std) )
        psis_ps.append(p)
        psis_Ds.append(D)
    psis_ps = np.array(psis_ps)
    psis_Ds = np.array(psis_Ds)
    
#    plt.hist(psis_all[1])
    
#    indx = np.argsort(ps)[-best_n:]
    indx = np.where((psis_ps >= min_psis_p) * (counts_ps >= min_counts_p))
    
#    for counts in psis_all[indx]:
#        plt.figure()
#        plt.hist(counts)
    df_t = df_t.iloc[indx]
    
    count_cvs =  sp.stats.variation(df_t.loc[:,(slice(None), "counts")].values, axis = 1)
    indx = np.where(count_cvs < counts_max_cv)
    
    
    return df_t.iloc[indx]
    
if __name__ == "__main__":
    None
    
#    anz = 200
#    psi_means = np.random.beta(4,4,size= anz)*0.4 + 0.3
#    counts = np.random.randint(1,10, anz)
#    tmp_compare_binomial_gillespie(counts,psi_means)
#    indx = np.where(counts == 4)
#    
#    sim = sim_tmp
#    i = indx[0][0]
#    sim.params = pars_tmp[i]
#    sim.results = results_tmp[i]
#    sim.plot_course(plot_psi=True)
#    
#    incl = sim.get_res_col("Incl")
#    skip = sim.get_res_col("Skip")
#    ges = incl + skip
#    fig = plt.figure()
#    plt.hist(incl + skip, bins = 20)
    ##s_res = s.compute_psi()
    ##s.plot_course()
    #
    #plt.scatter(counts, counts_tmp)
    #plt.scatter(psi_means, psis_tmp)