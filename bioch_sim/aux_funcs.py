#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:15:08 2019

@author: timur
"""
import os
from tkinter import filedialog
from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
import scipy as sp
import numba as nb
import numpy as np
import scipy.stats as st
from scipy.stats import gaussian_kde
import matplotlib
from tkinter import Tk 
import pickle



class empty(object):
    def __init__(self):
        pass

def plot_hist(x,title = "Distribution", bins = 15, ax = None, scale=1, exp_maxi=3, max_range=None ):
    if ax == None:
        fig, ax = plt.subplots(1, figsize=(5*scale,5*scale))
   
    x =  x[np.logical_not(np.isnan(x))]
    bins_count = bins
    sd, mean = np.std(x), np.mean(x)
    n, bins, patches = ax.hist(x, bins_count)
    #print(bins, mean, sd)
    y = st.norm.pdf(bins, mean, sd ) * (len(x)*(bins[1]-bins[0]))
    #ax.plot( bins, y, '--', lw=1, label='norm pdf')
    scores, coords, extrema = get_multimodal_scores(x, max_range, (len(x)*(bins[1]-bins[0])), exp_maxi)
    xx, y = coords[0], coords[1]
    ax.plot( xx, y, "--", color="red", label="KDE" )
    for i in extrema[0]:
        ax.annotate(' ', xy=(xx[i], y[i]), xytext=(xx[i], y[i]),
                    arrowprops=dict(facecolor='red', shrink=0.05),
        )
    for i in extrema[1]:
        ax.annotate(' ', xy=(xx[i], y[i]), xytext=(xx[i], y[i]),
                    arrowprops=dict(facecolor='green', shrink=0.05))
    
    ax.set_title(title + " (Bscore = %2.3f)" % scores[:,0].max(), fontsize=16)
#    ax.legend()
    return ax
def get_ind_from_maxi(x):
    maxi_ind = []
    for i in range(len(x)-2):
        j=i+1
        if x[j] > x[j-1] and x[j] >= x[j+1]:
            maxi_ind.append(j)
#    maxi_ind = np.array(maxi_ind)
#    return maxi_ind[np.argsort(x[maxi_ind])]
    return np.array(maxi_ind)
#@nb.njit
def get_multimodal_scores(x, max_range=None, scale = 1, exp_maxi=3):
    
    if len(x) < 1:
        return (np.array([[0,0,0]]), np.array(([0],[0])), ([0],[0]))
    mm = max(x) , min(x) 
    range_ = mm[0] - mm[1] 
#        print(range_)
    
    bw = range_/(exp_maxi*2) #doesnt work properly
    bw = "scott"
    xx = sp.linspace(mm[1]-range_*0.01, mm[0]+range_*0.01, 100)
    try:
        kde = gaussian_kde(x, bw_method = bw)
    except:
        return (np.array([[0,0,0]]), np.array(([0],[0])), ([0],[0]))
    y = kde.evaluate(xx) * scale
    scores_detailed = []
    if max_range == None:
        max_range = max(x) - min(x)
    maxi = get_ind_from_maxi(y)
    mini = get_ind_from_maxi(-y)
    m = max(y)
    for i in range(len(maxi)-1):
        for j in range(i+1, len(maxi)):
            
            min_min = min(y[k] for k in mini[i:j])
            m1 = y[maxi[i]] - min_min
            m2 = y[maxi[j]] - min_min
            m1, m2 = (min([m1,m2]), max([m1,m2]))
    #        print(m, m1,m2)
            d = abs(xx[maxi[i]]-xx[maxi[j]])
            
            width = d/max_range
#            print(d, max_range)
            height = m1/m2*m2/m
            sc = width*height
            
            sc *= 1 - kde.integrate_box_1d(xx[mini[i]], xx[mini[j-1]])
            
            scores_detailed.append([sc, width ,height])
        
#        print(y[maxi])
#        print(y[mini])
#        print((m1,m,d,r))
#    print(scores_detailed)
    if len(scores_detailed) == 0:
        scores_detailed.append([0,0,0])
    return (np.array(scores_detailed), np.array((xx, y)), (maxi, mini))

def get_bimodal_score(x, max_range=None, scale = 1, exp_maxi=3, tendency=False):
    x = x[~np.isnan(x)]
    tend = 0
    if tendency:
        tend = np.std(x)
    res = get_multimodal_scores(x, max_range=None, scale = 1, exp_maxi=3)
    return (res[0])[:,0].max() + tend


def sigmoid(x, mu=0, y_bounds=(0,1), range_95=6):
    y=1/(1+np.exp(6/range_95*(-x+mu)))
    y=y*(y_bounds[1]-y_bounds[0])+y_bounds[0]
    return y

@nb.njit(nb.f8[:,:](nb.f8[:,:],nb.f8[:]))
def rasterize(x, steps):
    # first col of x must contain time
    # steps is an array
    rows, cols = len(steps), x.shape[1]
    max_len = len(x)
    res = np.zeros((rows,cols))
    i=1
    j=0
    #print(steps)
    for step in steps:
        
        if step >= x[i-1,0] and step <= x[-1,0]:
            while  i < max_len and not(step >= x[i-1,0] and step < x[i,0]):
                i+=1
            if i==0:
                res[0] = x[0]
            else:
                #linear interpolation
#                factor = (step-x[i-1,0])/(x[i,0]-x[i-1,0])
#                res[j] = factor * (x[i]-x[i-1]) + x[i-1]
                res[j] = x[i-1]
        res[j,0] = step
        j+=1
            
    return res
        
    
def save_sims(sims, file_name=None):
    
    if file_name == None:
        root = Tk()
        root.withdraw()
        file_name = filedialog.asksaveasfilename()
        
    with open(file_name, "wb") as f:
        pickle.dump(sims, f)
    

def load_sims(file_name=None):
    if file_name == None:
        root = Tk()
        root.withdraw()
        file_name = filedialog.askopenfilename()
        
    with open(file_name, "rb") as f:
        sims= pickle.load(f)
    return sims

@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8))
def hill(x, Ka, n):
    if x > 0:
        return 1/(1 + (Ka/x)**n)
    return 0



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

@nb.njit
def norm_proximity(x, a , r=1 , p=2):
    dist = abs(x-a)
    if p != 0:
        return (1/(1+(dist/r)**p))
    else:
        return int(dist<=r)

@nb.njit
def asym_porximity(x, a, l = 2, r=1, p=4):
    dist = x-a
    if p == 0:
        if dist < 0:
            return int(-dist <= l)
        else:
            return int(dist <= r)
    else:
        if dist < 0:
            return (1/(1+(-dist/l)**p))
        else:
            return (1/(1+(dist/r)**p))