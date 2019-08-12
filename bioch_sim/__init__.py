# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:45:28 2019

@author: imb30
"""

from tkinter import Tk # copy to clipboard function
from tkinter import filedialog

import types
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import scipy.stats as st
from scipy.stats import gaussian_kde
import time
import math
from threading import Thread
from multiprocessing import Process, Manager, Queue
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import gridspec
from scipy.optimize import fsolve
from scipy.integrate import odeint
import pickle
import re
import numba as nb
from sklearn.cluster import KMeans, MeanShift, k_means

# inline displaying
#%matplotlib inline
# settings for plots
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['font.family'] = ['sans-serif']
line_width = 2
color_pre = 'dodgerblue'
color_Incl = 'darkorange'
color_Skip = "green"
color_phase_space = 'dimgrey'
color_initial_state = 'crimson'
color_steady_state = 'saddlebrown'


class SimParam(object):
    def __init__(self, name, time=200, discr_points= 200, params={}, init_state={}):
        self.name = str(name)
        self.runtime = time
        self.set_raster_count(discr_points)
        self.params=params
        
        self.init_state=init_state
        self.simulate_ODE = False
        self._rate_funcs =[]
        self._reactions =[]
        self.id=id(self)
        self._is_compiled = False
        self._dynamic_compile = False
        self._reset_results()
    def set_param(self, name, value):
        self.params[name] = value
        self._is_compiled = self._dynamic_compile
        self._reset_results()
    def _reset_results(self):
        self.bimodality = {}
        self.results={}
    def set_raster_count(self, discr_points):
        self.raster =  sp.linspace(0,self.runtime,int(discr_points))
    def get_all_params(self):
        params = {}
        params["constants"] = np.array(list(self.params.values()))
        params["init_state"] = np.array(list(self.init_state.values()))
        params["raster"] = np.array(self.raster)
        params["runtime"] = self.runtime
        params["id"] = id(self)
        return params
    def param_str(self, sep=", "):
        s = sep.join([k + "=" + "%2.3f" % v for k,v in self.params.items()])
        return s
    def set_runtime(self, time):
        self.runtime=time
        
    def set_state(self, state):
        self._state = state
    def add_reaction(self,rate_func, reaction={}, name = None):
        if name == None:
            name = "t%d" % (len(self._reactions) +1)  
        self._rate_funcs.append(rate_func)
        self._reactions.append(reaction)
        for name in reaction.keys():
            if not name in self.init_state.keys():
                self.init_state[name] = 0
        self._is_compiled = False
#    @nb.jit  # couse performance drops
    def get_rates(self, state=None):
        """ state must contain time as first element
        """
        if state is None:
            state=self._state
        return self._rates_function(state)
    def get_reacts(self):
        #returns reaction matrix
        return self._reacts
    
    def get_derivation(self):
        return self._reacts.transpose().dot(self.get_rates())
    
    def get_latex(self):
        res = "\\begin{align}\
            H=\\begin{pmatrix}\n"
        for rf in self._rate_funcs:
            res += rf + "\\\\\n"
        res += "\\end{pmatrix}\n"
        
        res += ", C= \\begin{pmatrix}\n"
        res += "\\\\\n".join([k for k,v in self.params.items()])
        res += "\\end{pmatrix}\n"
        res += "= \\begin{pmatrix}\n"
        res += "\\\\\n".join([str(v) for k,v in self.params.items()])
        res += "\\\\\n"
        res += "\\end{pmatrix},\\\\ \n"
        res += "M = \\begin{pmatrix}\n"
        res += "\\\\\n".join([k for k,v in self.init_state.items()])
        res += "\\end{pmatrix}\n"
        res += "= \\begin{pmatrix}\n"
        res += "\\\\\n".join([str(v) for k,v in self.init_state.items()])
        res += "\\\\\n"
        res += "\\end{pmatrix}\n"
        
        res += "\\end{align}\n"
        res = re.sub("_","\\\\textunderscore ", res)
        res = re.sub("\*","\\\\cdot ", res)
        r = Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(res)
        r.update() # now it stays on the clipboard after the window is closed
        r.destroy()
        return res
        
    
    def compile_system(self, dynamic = False):
        #create reaction matrix
        self._reacts = np.zeros(( len(self._reactions), len(self.init_state)), dtype=int )
        self._state = list(self.init_state.values())
        self._state.insert(0,0)
        self._state = np.array(self._state, dtype=np.float64)
        names = list(self.init_state.keys())
        i=0
        for react in self._reactions:
            for substance in react.keys():
                self._reacts[i, names.index(substance)] += react[substance]
            i+=1
        
        #create rates function
        self._r_s = np.zeros((len(self._rate_funcs),))
        func_str= "@nb.njit\n"
#        func_str= ""
        
        func_str+= "def _r_f_(st, constants = None):\n"
        func_str+= "\t_r_s=np.zeros(%d)\n" % len(self._r_s)
        func_str+= "\tt=st[0]\n"
        i=0
        for func in self._rate_funcs:
            func_str += "\t_r_s[%d] = " %i + func + "\n"
            i+=1
        
        i=0
        for name in self.params.keys():
            if(dynamic):
                func_str = re.sub("\\b" + name + "\\b", "constants[%d]" % i, func_str)
            else:
                func_str = re.sub("\\b" + name + "\\b", "%e" % self.params[name], func_str)
            i+=1
        i=1
        for name in self.init_state.keys():
            func_str = re.sub("\\b" + name + "\\b", "st[%d]" % i, func_str)
            i+=1
        func_str += "\treturn _r_s \n"
        func_str += "self._rates_function=_r_f_ \n"
#        print(func_str)
#        print(self.param_str())
        exec(func_str)
        self._is_compiled = True
        self._dynamic_compile = dynamic
#        self._rates_function = types.MethodType( self._rates_function, self )
        return func_str
    def simulate(self, ODE = False, ret_raw=False, max_steps = 1e7):
        if not self._is_compiled:
            self.compile_system()
        self._state = list(self.init_state.values())
        self._state.insert(0,0)
        self._state = np.array(self._state, dtype=np.float64)
        print("simulate " + self.param_str())
        results={}
        params = self.get_all_params()
        initial_state = params["init_state"]
        tt = params["raster"]
        self._constants = np.array(list(self.params.values()))
        sim_st = compute_stochastic_evolution(self.get_reacts(),
                                              self._state,
                                              nb.f4(self.runtime),
                                              self._rates_function,
                                              self._constants,
                                              np.array(tt, np.float32),
                                              nb.int64(max_steps))
        
        if ret_raw:
            results["stochastic"] = sim_st
#        sim_st_raster = rasterize(sim_st, tt)
        results["stoch_rastr"] = sim_st
        
        from numba import cuda
        
        #determinisitic simulation
        
        y_0 = initial_state
        if ODE or self.simulate_ODE:
            #numba make odent slower
#            sol_deterministic = odeint(get_ODE_delta,y_0,tt,
#                                       args = (np.array(self.get_reacts().transpose(), dtype=np.float64)
#                                       , self._rates_function))
            sol_deterministic = odeint(get_ODE_delta,y_0,tt,args = (self,))
            #add time column
            results["ODE"] = np.hstack((tt.reshape(len(tt),1),sol_deterministic))
        self.results=results
        self.results = results
        return results
    
    def simulate_cuda(self, ODE = False, ret_raw=False, max_steps = 1e7):
        
        return 0
    
    def set_result(self, name, res):
        self.results[name] = res
    def get_result(self, name):
        if name in self.results.keys():
            return self.results[name]
        return None
    def get_res_col(self, name, method = "stoch"):
       if name not in self.init_state:
           return None
       if method == "stoch":
           return self.get_result("stoch_rastr")[:, self.get_res_index(name)]
       return self.get_result("ODE")[:, self.get_res_index(name)]
        
    def get_psi_cv(self):
        psi = self.results["PSI"][1]
        sd, mean = np.std(psi), np.mean(psi)
        return sd/mean
    def get_psi_mean(self):
        if "PSI" not in self.results:
            self.compute_psi(ignore_fraction=0)
        return np.mean(self.results["PSI"][1])
    def get_res_index(self, name):
        return list(self.init_state.keys()).index(name) +1
    
    def compute_psi(self, products = ["Incl", "Skip"], solution="stoch_rastr",
                    ignore_extremes = False, ignore_fraction=0.1, recognize_threshold = 1):
        
#        print("compute psi...")
        sim_st_raster = self.results[solution]
        start_ind = int(len(sim_st_raster)*ignore_fraction)
        incl_counts = self.get_res_col(products[0])
        skip_counts = self.get_res_col(products[1])
        
        indices =  np.array(np.where(incl_counts + skip_counts >= recognize_threshold))
        if ignore_extremes:
            indices_extr = [np.where((incl_counts != 0) * (skip_counts != 0))]
            indices = np.intersect1d(indices, indices_extr)
        
        indices = indices[np.where(indices >= start_ind)]
#        print(len(incl_counts), len(skip_counts))
        
        incl_counts = incl_counts[indices]
        skip_counts = skip_counts[indices]
        psi = incl_counts/(incl_counts+skip_counts)
#        psi = psi[np.logical_not(np.isnan(psi))]
#        np.nan_to_num(psi, False)
        #result contains times and values arrays
        self.results["PSI"] = np.array((sim_st_raster[indices,0], psi))
        return indices, psi
    
    def get_bimodality(self, name = "PSI", ignore_extremes=False, ignore_fraction=0.1, recognize_threshold=1, with_tendency=False):
        
        settings = (name, ignore_extremes, ignore_fraction, recognize_threshold, with_tendency)
        
        if not hasattr(self, 'bimodality'):
            self.bimodality = {}
        
        if settings in self.bimodality:
            return self.bimodality[settings]
        print("compute bimodality....")
        max_range = None
        if name == "PSI":
            self.compute_psi(ignore_extremes=ignore_extremes, ignore_fraction=ignore_fraction,
                             recognize_threshold=recognize_threshold)
            inp = self.results["PSI"][1]
            max_range = 1
        else:
            inp = self.get_res_col(name)
        res = get_multimodal_scores(inp, max_range=max_range)
        self.bimodality[settings] = (res[0])[:,0].max()
        if(with_tendency and name == "PSI"):
            return self.bimodality[settings] + np.std(self.results["PSI"][1])
        return self.bimodality[settings]
    
    #aux for plotting
    def _get_indices_and_colors(self, products=[]):
        indices = []
        if not hasattr(self, "colors"):
            self.colors = cm.rainbow(np.linspace(0, 1, len(self.init_state)+1))
        if isinstance(products, str):
            return [self.get_res_index(products)], colors
        for name in products:
            index = self.get_res_index(name)
            indices.append(index)
        return indices, self.colors
        
    def plot(self, ax = None, res=["ODE","stoch"] , psi_hist = False, 
             scale = 1, products = [], line_width =1):
        plot_size=3
        subplots = 2
        if psi_hist:
            subplots = 3
        if ax == None:
            fig, ax = plt.subplots(1,subplots, figsize=(plot_size*subplots*scale,plot_size*scale))
        else:
            ax = ax.subplots(1,subplots, figsize=(plot_size*subplots*scale,plot_size*scale))
        
        if len(products) == 0:
            products = list(self.init_state.keys())
        indices, colors = self._get_indices_and_colors(products)
        self.plot_course(ax[0], res = res, products = products)
        
        st_res_rasterized = self.get_result("stoch_rastr")
        
        cvs = ["CV(%s):%2.3f" % (p, sp.stats.variation(self.get_res_col(p))) for p in products]
        cvs = ", ".join(cvs)
        ax[1].hist(st_res_rasterized[:,np.array(indices)],label = tuple(products),
          orientation = "horizontal", color = tuple(colors[i-1] for i in indices))
        ax[1].set_title(cvs, fontsize=10)
        ax[1].set_xlabel("#")
        ax[0].legend( framealpha=0.7, loc="best", fontsize=10, ncol = len(products))
        ax[1].legend( framealpha=0.7, loc = "best", fontsize=10)
        
        if psi_hist:
            
            psi_res = self.compute_psi()[1]
            plot_hist(psi_res, ax=ax[2])
            
            #print((mean, sd))
            ax[2].set_title("PSI distribution (CV = %2.3f)" % self.get_psi_cv())
        #fig.legend()
        return fig, ax
    
    def plot_course(self, ax = None, res=["ODE","stoch"], products=[], rng = None,
                    line_width=2, scale = 1, plot_mean=False):
        if ax == None:
            fig, ax = plt.subplots(1, figsize=(10*scale,10*scale))
        
        tt = self.raster[:rng]
        #st_res =self.get_result("stochastic")
        stoch_res = self.get_result("stoch_rastr")
        ode_res = self.get_result("ODE")
       
        lines = []
        if len(products) == 0:
            products = list(self.init_state.keys())
        indices, colors = self._get_indices_and_colors(products)
        
        for index in indices:
            name = list(self.init_state.keys())[index-1]
            index = self.get_res_index(name)
            color = self.colors[index-1]
            if "stoch" in res:
                lines.append(ax.plot(stoch_res[:rng,0],stoch_res[:rng,index], label = name +"(stoch)",
                         color = color, lw = 0.5*line_width,drawstyle = 'steps')[0])
                
            mean = np.mean(stoch_res[int(len(stoch_res)/3):,index])
            #plot mean of stoch sim
            if plot_mean:
                ax.plot([0,tt[-1]], [mean,mean], "--", color=color, lw=line_width)
            if "ODE" in res and ode_res is not None:
                ax.plot(tt,ode_res[:rng,index],"--", color = color, lw = 1.5*line_width, label = name + "(ODE)")
        
        #ax.yaxis.set_label_coords(-0.28,0.25)
        ax.set_ylabel(self.param_str("\n"), rotation=0, fontsize="large" )
        ax.set_ylabel("#")
        ax.set_xlabel("time",fontsize="large" )
        ax.set_title(self.name)
        ax.legend()
        return ax

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
#        print(particular_scores)
#        m=max(n)
#        score, pos = bimodality_score(x, bins_count)
#        ax.plot([pos[0],pos[0]], [0,m])
#        ax.plot( [pos[1],pos[1]], [0,m])
    #print((mean, sd))
    
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
    tend = 0
    if tendency:
        tend = np.std(x)
    res = get_multimodal_scores(x, max_range=None, scale = 1, exp_maxi=3)
    return (res[0])[:,0].max() + tend
    
#@nb.njit#(nb.f8[:](nb.f8[:], nb.f8, nb.f8[:,:], nb.f8[:](nb.f8[:])))        
#def get_ODE_delta(x,t,reacts, rate_func):
#   
#    x_arr= np.zeros(len(x)+1)
#    x_arr[1:] = x
#    x_arr[0] = t
#    rates = rate_func(x_arr)
#    
#    dx = reacts.dot(rates)
#    return dx

def get_ODE_delta(x,t,sim):
   
    x_arr= np.zeros(len(x)+1)
    x_arr[1:] = x
    x_arr[0] = t
    #print(t)
    reacts = sim.get_reacts()
    rates = sim.get_rates(x_arr)
#    print(rates)
    dx = reacts.transpose().dot(rates)
    return dx
  
@nb.njit#(nb.f8(nb.f8[:,:], nb.f8[:], nb.f4, nb.f8[:](nb.f8[:],), nb.i4))
def compute_stochastic_evolution(reactions, state, runtime, rate_func, constants, time_steps, max_steps):
   
    STATE = np.zeros((len(time_steps), len(state)),dtype=np.float64)
#    print(state.dtype)
#    state= np.array(state, dtype = np.int64)
#    STATE[0,:] = state
    tf = runtime
    tt = 0
    steps = nb.int64(0)
    i = steps
    length = len(time_steps)
    while tt <= tf and steps < max_steps and i < length:
        
        # sample two random numbers uniformly between 0 and 1
        rr = sp.random.uniform(0,1,2)
        a_s = rate_func(state, constants)
#        print(a_s)
        a_0 = a_s.sum()
        # time step: Meine Version
        #print(a_0)
        if a_0 == 0:
            break
        tt = tt - np.log(1. - rr[0])/a_0
        state[0] = tt
        if(tt >= time_steps[i]):
            STATE[i,:] = state
            STATE[i:0] = time_steps[i]
            i+=1
        # find the next reaction
        prop = rr[1] * a_0
        cum_a_s = np.cumsum(a_s)
        #print(cum_a_s, prop)
        ind = np.where(prop <= cum_a_s)[0][0]
        # update the systems state
        state[1:] +=reactions[ind]
        steps+=1
#        STATE[steps,:] = state
        
    
#    return STATE[0:steps+1]
    return STATE
# define parameters  

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
        
def simulate(sim, ODE = True, ret_raw=False, max_steps = 1e7):
    return sim.simulate(ODE = ODE, ret_raw=ret_raw, max_steps = max_steps)
   
#unfinished(useless) function
def plot_psi_to_cv(sims):
    scale = 2
    fig, ax = plt.subplots(1, figsize=(5*scale,5*scale))
    colors = cm.rainbow(np.linspace(0, 1, len(sims)))
    for sims_row, col in zip(sims, colors) :
        cvs = [s.get_psi_cv() for s in sims_row]
        psis = [s.get_psi_mean() for s in sims_row]
#        psis = [s.expected_psi for s in sims_row]
        
        sds = [np.std(s.results["PSI"][1]) for s in sims_row]
        
        ax.plot(psis, cvs, label = "v_syn: %2.2f" % sims_row[0].params["v_syn"], color = col)
        #ax.plot(psis, sds, "--", color = col)
        ax.set_ylabel("CV(psi)")
        ax.set_xlabel("mean(psi)")
        ax.legend()

def plot_sd_over_t(sims, value = "PSI"):
    scale = 2
    fig, ax = plt.subplots(1, figsize=(5*scale,5*scale))
    colors = cm.rainbow(np.linspace(0, 1, len(sims)))
    for sim, col in zip(sims, colors) :
        psi = sim.results[value]
        indices = range(10,len(psi), 5)
        sds = [np.std(psi[0:l]) for l in indices]
#        ax.plot(indices, sds, label = "v_syn: %2.2f" % sim.params["k_syn"], color = col)
        ax.plot(indices, sds, label = sim.param_str()[:10], color = col)
#        ax.plot(indices, psi[indices], "--", color=col, )
        
        ax.set_ylabel("SD (psi)")
        ax.set_xlabel("sim duration")
        ax.set_title("Stability over duration")
        #ax.legend()
    lgd =ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.tight_layout(pad=7)

def bimodality_score(x, bins = 20):
    mean = np.median(x)
    x, bins = np.histogram(x, bins)
    
    i=0
    while mean > bins[i]:
        i+=1
    i-=1
    j=i
    max_ind = [i,j]
    while i < len(x):
        if x[i]> x[max_ind[0]]:
            max_ind[0]=i
        i+=1
    while j >= 0:
        if x[j]> x[max_ind[1]]:
            max_ind[1]=j
        j-=1
    pos = [(bins[max_ind[0]+1] + bins[max_ind[0]])/2,
           (bins[max_ind[1]+1] + bins[max_ind[1]])/2]
    score = (pos[0] - pos[1])/mean
    return score, pos

def run_sims(sims, proc_count = 1):
    
    q1 = Queue()
    q2 = Queue()
    qerr = Queue()
    procs =[]
    for i in range(0,proc_count):
        print("creating process %d" % i )
        p = Process(target = child_process_func, args=(q1,q2,qerr))
        procs.append(p)
        p.start()
        print("Process %d started" % i)
    ids_to_sim = {}
    for sim in sims:
        print("Putting sim %s in the queue" % sim.name)
        q1.put(sim)
        ids_to_sim[id(sim)] = sim
    
    # send end signal to child processes
    for p in procs:
        q1.put(False)
    
    #get results
    try:
        for i in sims:
            res = q2.get(timeout=30)
            sim = ids_to_sim[res["id"]] 
            sim.results = res
            print("Results acquired for %s\n %s" % (sim.name, sim.param_str()))
    except RuntimeError as err:
        print(err)
        print(qerr.get(timeout=1))
    finally:
        [p.terminate() for p in procs]
        [p.join() for p in procs]
    
    

def child_process_func(q1,q2, qerr):
    
    try:
        while True:
            sim = q1.get()
            if sim==False:
                break
            res=sim.simulate()
            res["id"] = sim.id
            q2.put(res)
            del sim
    except RuntimeError as err:
        qerr.put(err)
        
        

def plot_3d(sims, value="PSI", axes = ("v_syn", "s1")):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    x = np.zeros((len(sims), len(sims[0])))
    y = np.copy(x)
    z = np.copy(x)
    
    i = 0
    for ss in sims:
        j=0
        for s in ss:
            x[i,j] = s.params[axes[0]]
            y[i,j] = s.params[axes[1]]
            if value == "CV":
                z[i,j] = s.get_psi_cv()
            else:
                z[i,j] = s.get_psi_mean()
            j+=1
        i+=1
    
    #ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    ax.plot_surface(x,y,z,  cmap=cm.coolwarm)
    #ax.plot_wireframe(x,y,z,  cmap=cm.coolwarm)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_zlabel(value)
    
    plt.show()
    
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