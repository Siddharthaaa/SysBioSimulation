# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:06:24 2019

@author: timuhorn
"""
"""
Title:
    auxiliary functions
    
"""

import pylab as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
from bioch_sim import *
from sklearn.linear_model import LinearRegression

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

def show_splicing_data(df=None, ax=None, best_n = 20, min_psi_th = 0.3):
    
    
    ness_cols = ["ID", "Bscore", "mean", "std", "PSI_values"]
    
    for ness_col in ness_cols:
        if ness_col not in df:
            raise ValueError("DataFrame must contain following columns:", ness_col)
    
    
    if ax == None:
        fig, ax = plt.subplots(1,3, squeeze= True)
#    df.set_index("ID", inplace = True)
    sorted_df = df.sort_values("Bscore", ascending = False)
    
    if(best_n == None):
        best_n = len(np.argwhere(df["Bscore"] >= min_psi_th))
    
    ax[0].plot(df["mean"], df["std"], ".", c = "blue", label="All PSI values")
    ax[0].set_title("SD over PSI")
    ax[0].set_xlabel("mean(PSI)")
    ax[0].set_ylabel("SD(PSI)")
    means = sorted_df.iloc[:best_n, list(df).index("mean")]
    stds = sorted_df.iloc[:best_n, list(df).index("std")]
    names  = sorted_df.iloc[:best_n, list(df).index("ID")]
    ax[0].plot(means, stds, ".", c = "red", label="High Bscore")
    ax[0].legend()
    
    ax[1].hist(df["Bscore"])
    ax[1].set_title("Bscore distribution")
    ax[1].set_xlabel("Bscore")
    
    plot_hist(sorted_df.iloc[0, list(df).index("PSI_values")], ax= ax[2])
    ax[2].set_title("Best Bimodality: %s" % sorted_df.iloc[0, list(df).index("ID")])
    ax[2].set_xlabel("PSI")
    ax[2].set_ylabel("cell count")
    
    
    all_val = np.array([a for a in sorted_df.iloc[:best_n, list(sorted_df).index("PSI_values")].values])
    df_bimodal_as = pd.DataFrame(all_val).transpose()
    #heatmap
#    im = ax[3].imshow(df_bimodal_as.corr())
#    cbar = ax[3].figure.colorbar(im, ax=ax[3])
#    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    
    
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
    if("tpm" in list(df)):
        intens = np.array([a for a in sorted_df.iloc[:best_n, list(sorted_df).index("tpm")].values])
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
            ax.set_ylabel("TPM", fontsize =10)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            
    if("tpm" in list(df) and "counts" in list(df)):
        tpms = np.array([a for a in sorted_df.iloc[:best_n, list(sorted_df).index("tpm")].values])
        counts = np.array([a for a in sorted_df.iloc[:best_n, list(sorted_df).index("counts")].values])
        fig = plt.figure()
        for i in range(best_n):
            ax = plt.subplot(nrows, ncols, i+1)
            ax.plot(counts[i], tpms[i],".")
            ax.set_title(names[i])
            ax.set_xlabel("counts")
            ax.set_ylabel("tpms")    
    
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
        ax.hist([tpms[x], tpms[y]], label = [names[x], names[y]])
        ax.legend(fontsize = 10)
        ax.set_xlabel("tpm")
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        
        ax = plt.subplot(first_n, 3, j+2)
        ax.plot(all_val[x], tpms[x], ".", label = names[x] )
        ax.plot(all_val[y], tpms[y], ".", label = names[y] )
        ax.set_ylabel("TPM", fontsize = 10)
        ax.set_xlabel("PSI", fontsize = 10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.legend(fontsize = 10)
        ax.set_title("")
        j+=3
    
    
    indx_pairs = np.argsort(df_corr)
    return (sorted_df, indx_pairs, df_corr)


def fitting_distribution(sim_obj = None, distr = None, param_bounds = {}):
    return None

def general_analysis(df = None, ax = None, th_suppoints = 10):
    fig = plt.figure()
    ax = plt.subplot(1,3,1)
    stds = df.loc[:,"std"].values
    means = df.loc[:,"mean"].values
    intensity = np.array([np.array(a, dtype=np.float64) for a in df.loc[:,"counts"].values])
    intensity = np.array([np.mean(a[np.logical_not(np.isnan(a))]) for a in intensity])
    intensity = np.log(1+ intensity)
#    intensity = intensity / np.max(intensity[np.logical_not(np.isnan(intensity))])
    colors = cm.rainbow(intensity)
#    print(colors)
    paths = ax.scatter(means, stds, c=intensity )
    cbar = fig.colorbar(paths, ax = ax)
#    cbar = ax.figure.colorbar(None, ax=ax)
    cbar.ax.set_ylabel("log (mean(counts) + 1)", rotation=-90, va="bottom")
    ax.set_xlabel("mean(PSI)")
    ax.set_ylabel("SD(PSI)")
    
    
    ax = plt.subplot(1,3,2)
    
    all_val = np.array([a for a in df.iloc[:, list(df).index("PSI_values")].values])
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
    intens = np.array([a for a in df.iloc[:, list(df).index("counts")].values])
    slopes = []
    for i in range(len(all_val)):
        if(intens[i] is not None):
            t_inds = np.where(~np.isnan(all_val[i]) & ~np.isnan(intens[i]))
#            print(t_inds)
            if(len(t_inds[0]) >= th_suppoints):
                x = np.array(all_val[i][t_inds])
                y = np.log(1+intens[i].values[t_inds])
                model = LinearRegression()
        #            print(x.reshape((-1, 1)))
                model.fit(x.reshape((-1, 1)), y)
                slopes.append(model.coef_[0])
                
    ax.hist(slopes, bins=100)
    ax.set_title("Slope of LinReg btw. PSI and log(1+counts)")
    ax.set_ylabel("#")
    ax.set_xlabel("slope")
    return (intensity, df_corr, slopes)

#plt.plot(res[sort[-20]].all_val, res[sort[-18]].all_val, ".")
#plt.plot(df_bimodal_as.iloc[:,0], df_bimodal_as.iloc[:,6], ".")


#test function for validation of optimization alg.

def _tmp_plot_psi_to_intens(df = None, log = True, th_suppoints = 10 ):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    stds = df.loc[:,"std"].values
    means = df.loc[:,"mean"].values
    intensity = np.array([np.array(a, dtype=np.float64) for a in df.loc[:,"counts"].values])
    intensity = np.array([np.mean(a[np.logical_not(np.isnan(a))]) for a in intensity])
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

def rastrigin(x, A = 10):
    l = len(x)
    
    return l*A + np.array([(a*a - A*np.cos(2*np.pi*a)) for a in x]).sum(axis=0)

def create_DataFrame_template():
    df = pd.DataFrame()
    df['PSI_values'] = np.nan
    df['PSI_values'] = df['PSI_values'].astype(object)
    df['tpm'] = np.nan
    df['tpm'] = df['tpm'].astype(object)
    df['counts'] = np.nan
    df['counts'] = df['counts'].astype(object)
    
    return df