# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:04:04 2019

@author: Timur
"""

import pylab as plt
import numpy as np
import scipy as sp
import scipy.stats as st

import time
import math

import matplotlib
import matplotlib.colors as colors
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib.text import Text
from matplotlib import gridspec
from scipy.optimize import fsolve

from .aux_funcs import *

class SimPlotting:
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
    
    def plot_course(self, ax = None, res=["ODE","stoch"], products=[], products2=[],
                    t_bounds = (0, np.inf), line_width=2, scale = 1, plot_mean=False,
                    plot_psi=False, clear = False):
        if ax == None:
            fig, ax = plt.subplots(1, figsize=(10*scale,10*scale))
        ax2 = None
        if type(products) == str:
            products = products.split(" ")
        indx = np.where((self.raster >= t_bounds[0]) * (self.raster <= t_bounds[1]))[0]
        tt = self.raster[indx]
        #st_res =self.get_result("stochastic")
        stoch_res = self.get_result("stoch_rastr")
        ode_res = self.get_result("ODE")
        if len(products) == 0:
            products = list(self.init_state.keys())
        #separate Species and Expressions
        exprs = []
        for prod in products.copy():
            if prod not in self.init_state:
                exprs.append(prod)
                products.remove(prod)
    
        lines = []
        indices, colors = self._get_indices_and_colors(products)
        # plot species
        for index in indices:
            name = list(self.init_state.keys())[index-1]
            index = self.get_res_index(name)
            color = self.colors[index-1]
            if "stoch" in res:
                lines.append(ax.plot(stoch_res[indx,0],stoch_res[indx,index], label = name,
                         color = color, lw = 0.5*line_width, drawstyle = 'steps-post')[0])
                
            mean = np.mean(stoch_res[int(len(stoch_res)/3):,index])
            #plot mean of stoch sim
            if plot_mean:
                ax.plot([tt[0],tt[-1]], [mean,mean], "--", color=color, lw=line_width)
            if "ODE" in res and ode_res is not None:
                ax.plot(tt,ode_res[indx,index],"-", color = color, lw = 1.5*line_width, label = name + "(ODE)")
        
        #plot expressions
        for i, e in enumerate(exprs):
            res = self.get_res_by_expr(e, t_bounds = t_bounds)
            ax.plot(tt,res, "-.", color ="C"+str(i), lw=0.7*line_width, label =e)
        
        #ax.yaxis.set_label_coords(-0.28,0.25)
        if len(products2)>0:
            if type(products2) is str:
                products2 = [products2]
            ax2 = ax.twinx()
            self.plot_course(ax = ax2, res=res, products=products2, t_bounds = t_bounds, clear=True)
#            ax_2.legend()
        if(plot_psi):
            ax_psi = ax.twinx()
            (indx, psis) = self.compute_psi()
            ax_psi.plot(self.raster[indx], psis, ".", markersize=line_width*5, label = "PSI")
            ax_psi.set_ylabel("PSI")
        if(not clear):
            ax.set_ylabel(self.param_str("\n"), rotation=0, fontsize="large" )
    #        ax.set_ylabel("#")
            ax.set_xlabel("time",fontsize="large" )
            ax.set_title(self.name)
            ax.legend()
        return ax, ax2
    
    def plot_series(self, ax=None, products=[], t_bounds=(0, np.inf), scale=1):
        if ax == None:
            fig, ax = plt.subplots(1, figsize=(10*scale,5*scale))
        if type(products) == str:
            products = products.split(" ")
        if len(products) == 0:
            products = list(self.init_state.keys())
            
        print(products)  
        indx = np.where((self.raster >= t_bounds[0]) * (self.raster <= t_bounds[1]))[0]
        tt = self.raster[indx]
        print(tt)
        indices, cols = self._get_indices_and_colors(products)

        for index, col, p in zip(indices, cols, products):
            results = self._last_results[:,indx,index]
            mean = np.mean(results, axis = 0)
            ax.plot(tt, results.T, c=col, lw=0.2, alpha=0.3, antialiased=True)
            ax.plot(tt, mean, c = col, lw=3, label = p, alpha=1)
#        ax.legend()
        count = len(self._last_results)
        ax.set_title(str(count) + " Realizations")
        ax.set_xlabel("time")
        ax.set_ylabel("#")
        return ax
    
    def plot_parameters(self, parnames =[], parnames2=[], annotate=False, ax=None, **plotargs):
        
        if ax == None:
            fig, ax = plt.subplots(1, figsize=(10,3))
        constants = self._evaluate_pars()
        chd_pars = []
        t_events = sorted(self._time_events.copy())
        values = []
        ts = [] #time points
        ts.append(0)
        values.append(list(constants.values()))
        for k, te in enumerate(t_events):
            if(te.t <= self.runtime):
                ts.append(te.t)
                chd_pars += te.apply_action(constants)
                values.append(list(constants.values()))
                if(annotate):
                    ax.annotate(te.name, (te.t,0),  (-40, -(40+k*20)),
                                textcoords = "offset pixels", 
                                arrowprops={"arrowstyle": "-"})
        ts.append(self.runtime)
        values.append(list(constants.values()))
        values = np.array(values)
        if len(parnames) < 1:
            parnames = chd_pars
        indx = np.where([k  in parnames for k in constants])[0]
#        values = values[:,indx]
#        print(ts)
#        print(values)
#        print(parnames)
        for ix, i in enumerate(indx):
            print(i)
            print(values[:,i])
            ax.plot(ts, values[:,i],
                    label = list(constants)[i],
                    drawstyle = 'steps-post',
                    **plotargs)
        ax.legend(loc="upper left")
        if len(parnames2)>0:
            ax2 = ax.twinx()
            self.plot_parameters(parnames=parnames2, annotate=False, ax=ax2)
            ax2.legend(loc="upper right")
            
        return ax
    
    def annotate_timeEvents(self, ax = None, offset=20):
        t_events = sorted(self._time_events.copy())
        ts = [] #time points
        ts.append(0)
        for k, te in enumerate(t_events):
            if(te.t <= self.runtime):
                ts.append(te.t)
                ax.annotate(te.name, (te.t,0),  (-40, -(40+k*offset)),
                            textcoords = "offset pixels", 
                            arrowprops={"arrowstyle": "-"})
    def annotate_timeEvents2(self, ax = None, y_axes_offset=0, text_ax_offset=-0.2,
                             line_kw=dict(),
                             text_kw=dict()):
        t_events = sorted(self._time_events.copy())
        ts = [] #time points
        ts.append(0)
        te_textes = []
        fig = ax.get_figure()
        inv = ax.transData.inverted()
        _tmp, y_offset = ax.transAxes.transform((0, y_axes_offset))
        print("offset coord ", (y_offset))
        trans = transforms.blended_transform_factory(
                    ax.transData, ax.transAxes)
        renderer = fig.canvas.get_renderer()
        for k, te in enumerate(t_events):
            if(te.t <= self.runtime):
                ts.append(te.t)
                text = ax.annotate(te.name, (te.t, y_axes_offset),  (0, text_ax_offset),
                            xycoords = trans,
                            textcoords = ("offset pixels","axes fraction"), 
                            annotation_clip = False,
                            arrowprops={"arrowstyle": "-"},
                            **text_kw)
                text.arrow_patch.set_color("grey")
                text.arrow_patch.set_ls("--")
                
                text.set_va("center")
                text.set_ha("center")
                bbox = text.get_window_extent(renderer = renderer)
                x0, y0, w, h = inv.transform(bbox.bounds)
                print("AAAAAAAA coord: \n", bbox.bounds)
                text_ok = False
                i = 1
                while(not text_ok):
                    text.set_position((0, text_ax_offset - i*0.01))
                    bbox = text.__class__.__bases__[0].get_window_extent(text, renderer = renderer)
#                    print("AAAAAAAAAAAAA", bbox.bounds)
                    i+=1
                    text_ok = True
                    for t in te_textes:
                        bb = t.get_window_extent()
                        if(bb.fully_overlaps(bbox)):
                            text_ok = False
                te_textes.append(text)
                x0, y0, w, h = inv.transform(bbox.bounds)
#                ax.plot((te.t, te.t), (offset, 10), ls="-", clip_on=False)    
    
    
    def plot_par_var_1d(self, par = "s1", vals = [1,2,3,4,5], label = None, label_axes=False,
                        ax=None, plot_args = dict(), func=None, **func_pars):
        res = []
        last_results = []
        for v in vals:
            self.set_param(par, v)
            self.simulate()
            last_results.append(self._last_results[0])
            if func is not None:
                res.append(func(**func_pars))
        self._last_results = np.array(last_results)
        if ax is None:
            fig, ax = plt.subplots()
        
        if (len(res) > 0):
            ax.plot(vals, res, label = label, **plot_args)
            if(label_axes):
                ax.set_ylabel(func.__func__.__name__ +   str(func_pars))
                ax.set_xlabel(par)
        
        return ax
    
    def plot_par_var_2d(self, pars = {"s1":[1,2,3], "s2": [1,2,4]},ax = None,
                                      plot_args = dict(), func=None, **func_pars):
        
        names = list(pars.keys())
        params = list(pars.values())
        print(params)
        
        if(type(names[0]) is str):
            names[0] = [names[0]]
        if(type(names[1]) is str):
            names[1] = [names[1]]
        
        res = []
        sim_res =[]
        for par1 in params[0]:
            r = []
            sim_r = []
            for n in names[0]:    
                self.set_param(n, par1)
            for par2 in params[1]:
                
                for n in names[1]:    
                    self.set_param(n, par2)
                self.simulate()
                sim_r.append(self.results["stoch_rastr"])
                r.append(func(**func_pars))
            res.append(r)
            sim_res.append(sim_r)
            
        if ax is None:
            fig, ax = plt.subplots()
        
        heatmap(np.array(res), params[0], params[1], ax,
                cbarlabel= func.__func__.__name__ +   str(func_pars),
                **plot_args)
        ax.set_xlabel(", ".join(names[1]))
        ax.set_ylabel(", ".join(names[0]))
        
        results = empty()
        results.ax = ax
        results.values = res
        results.sim_res = sim_res
        return results
