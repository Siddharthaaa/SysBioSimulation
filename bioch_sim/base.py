# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:45:28 2019

@author: imb30
"""


from tkinter import Tk # copy to clipboard function


import types
import pylab as plt

import numpy as np
import scipy as sp
import scipy.stats as st

import time
import math

import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import gridspec
from scipy.optimize import fsolve
from scipy.integrate import odeint
#other ode solver. IMPORTANT
#from scikits.odes import ode

import re
import numba as nb
import numba.typed
#from numba.typed import List
#from sklearn.cluster import KMeans, MeanShift, k_means

import random

from .timeevent import TimeEvent
from .interface import *
from .aux_funcs import *

# inline displaying
#%matplotlib inline

class empty(object):
    def __init__(self):
        pass

class SimParam():
    def __init__(self, name, t=200, discr_points= 1001, params={}, init_state={}):
        self.name = str(name)
        self.runtime = t
        self.set_raster(discr_points)
        self.params=params
        
        self.init_state=init_state
        self.simulate_ODE = False
        self._rate_funcs =[]
        self._reactions =[]
        self._transitions = {}
        self.id=id(self)
        self._is_compiled = False
        self._dynamic_compile = False
        self._clusters ={}
        self._reset_results()
        self._time_events = []
    def set_param(self, name, value):
        self.params[name] = value
        self._is_compiled = self._dynamic_compile
        self._reset_results()
    
    def _reset_results(self):
        self.bimodality = {}    
        self.results={}
        self._constants = np.array(list(self.params.values()))
    def set_raster(self, discr_points=None):
        if(discr_points is None):
            discr_points = self.raster_count
        self.raster_count = discr_points
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
        s = sep.join([k + "=" + "%e" % v for k,v in self.params.items()])
        return s
    def set_runtime(self, t):
        self.runtime=t
        self.set_raster(self.raster_count)
        
    def set_state(self, state):
        self._state = state
    def add_reaction(self,rate_func, reaction={}, name = ""):
        name = ("t%d: " % (len(self._reactions) +1)) + name
        self._rate_funcs.append(rate_func)
        self._reactions.append(reaction)
        #all elements should be lists
        for k in reaction.keys():
            if type(reaction[k]) is not list:
                reaction[k] = [reaction[k]]
        
        self._transitions[name] = dict(rate=rate_func,
                                      actors = reaction)
        
        for name in reaction.keys():
            if not name in self.init_state.keys():
                self.init_state[name] = 0
        self._is_compiled = False
#    @nb.jit  # causes performance drops
        
    def add_timeEvent(self, te):
        te.set_constants(self.params)
        self._time_events.append(te)
#        self._time_events = sorted(self._time_events)
    def delete_timeEvents(self):
        self._time_events = []
    def get_rates(self, state=None):
        """ state must contain time as first element
        """
        if state is None:
            state=self._state
        self._update_pre(state, self._constants, self._curr_pre)
        return self._rates_function(state, self._constants
                                    , np.zeros(len(self._reactions)),
                                    self._curr_pre)
    def get_reacts(self, state = None):
        #returns reaction matrix
        if state is None:
            state = self._state
        self._update_pre(state, self._constants, self._curr_pre)
        self._update_post(state, self._constants,self._curr_post)
        self._reacts= self._curr_post - self._curr_pre
        return self._reacts
    def get_derivation(self, state = None):
        return self.get_reacts(state).transpose().dot(self.get_rates())
    def set_cluster(self, name, c = ()):
        self._clusters[name]= c
    def get_latex(self):
        
#        s = "expr"
#        lat = sympy.latex(sympy.sympify(s))
        
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
    
    def show_interface(self):
        root = Tk()
        root.geometry("1000x600+300+300")
        if "stoch_rastr" not in self.results:
            self.simulate()
        app = SimInterface(self)
        root.mainloop()
        
    
    def compile_system(self, dynamic = True, add_ns={}):
        add_ns = add_ns.copy()
        #convert function to jitted funcitons
        for k, v in add_ns.items():
            if(type(v) is types.FunctionType):
                add_ns[k] = nb.njit(v)
        globs = globals()
        globs.update(add_ns)
                
        self._constants = np.array(list(self.params.values()))
        #create reaction matrix
        n = len(self._reactions)
        m = len(self.init_state)
        self._reacts = np.zeros((n,m ), dtype=int )
        self._pre = np.zeros((n,m ), dtype = object)
        self._post = self._pre.copy()
        self._state = list(self.init_state.values())
        self._state.insert(0,0)
        self._state = np.array(self._state, dtype=np.float64)
        names = list(self.init_state.keys())
    
        self._fire_transition =[]
        self._fire_transition_str =[]
        func_update_pre = "@nb.njit\ndef update_pre(st, pars, pre):\n"
        func_update_post = "@nb.njit\ndef update_post(st, pars, post):\n"
        for i, react in enumerate(self._reactions):
            update_st_func = "@nb.njit\ndef update_st(st, pars):\n"
            subs_to_update = []
            updates = []
            for substance in react.keys():
                subs_i = names.index(substance)
                subs_to_update.append("st[%d]" % (subs_i+1))
                update = ""
                for v in react[substance]:
                    if(v is None):
                        #create inhibition 
                        v = "-2*" + substance
                    elif(v == 0):
                        # create flush
                        v = "-" + substance
                    update += ("" if update == "" else " +") + str(v)
                    if type(v) is str:
                        if v.startswith("-"):
                            v = v[1:]
                            self._pre[i, subs_i] = v
                            func_update_pre += "\tpre[%d,%d] = %s\n" % (i, subs_i, v)
                        else:
                            self._post[i, subs_i] = v
                            func_update_post += "\tpost[%d,%d] = %s\n" % (i, subs_i, v)
                    else:
                        if v < 0:
                            self._pre[i, subs_i] += -v
                        elif v > 0:
                            self._post[i, subs_i] += v
                        
#                self._reacts[i, names.index(substance)] += sum(react[substance])
                update = self._sub_vars(update)
                updates.append(update)
#            update_st_func += "\t" + " ,".join(subs_to_update) + " +=" + " ,".join(updates) + "\n"
            update_st_func += "\n".join(["\t" + subs + " += " + update for subs, update in zip(subs_to_update, updates)])
            update_st_func += "\n\treturn None \nself._fire_transition.append(update_st)\n"
            self._fire_transition_str.append(update_st_func)
            #TODO str contains BUG potential
            exec(update_st_func)
        func_update_pre += "\treturn None \nself._update_pre = update_pre"
        func_update_post += "\treturn None \nself._update_post = update_post"
        func_update_post = self._sub_vars(func_update_post)
        func_update_pre = self._sub_vars(func_update_pre)
        print(func_update_post)
        print(func_update_pre)
        exec(func_update_post)
        exec(func_update_pre)
#        self._reacts = self._post - self._pre
        
        self._compile_timeEvents()
        
        #create rates function
        self._r_s = np.zeros((len(self._rate_funcs),))
        func_str= "@nb.njit\n"
#        func_str= ""
        
        func_str+= "def _r_f_(st, pars, _r_s, _pre):\n"
#        func_str+= "\t_r_s=np.zeros(%d)\n" % len(self._r_s)
        func_str+= "\tt=st[0]\n"
        for i, (func, tr) in enumerate(zip(self._rate_funcs, self._transitions.values())):
            v_to_chek =[]
            for j, v in enumerate(self._pre[i]):
                if(v != 0):
#                    print(v)
#                    v_to_chek.append(str(v) + " <= st[%d]" % (j+1))
                    v_to_chek.append("_pre[%d,%d] " % (i,j) + " <= %s" % list(self.init_state)[j])
            
            if len(v_to_chek) > 0:
                func = "(" + func + ") if (" + ") and (".join(v_to_chek) + ") else 0" + "\n"
#                func_str += "\t_r_s[%d] = (" %i + func + ") if (" + ") and (".join(v_to_chek) + ") else 0" + "\n"
#            self._rate_funcs_extended.append(func)
            tr["rate_ex"] = func
            func_str += "\t_r_s[%d] = " %i + func  + "\n"
        print(func_str)
        func_str = self._sub_vars(func_str, par_name = "pars", place_name="st", dynamic=dynamic)
        func_str += "\treturn _r_s \n"
        func_str += "self._rates_function=_r_f_ \n"
#        print(func_str)
#        print(self.param_str())
#        locs = locals()
#        locs.update(add_ns)
#        print(locs)
#        exec(func_str, globals(),locs)
        exec(func_str)
#        self._update_st_funcs =[]
        self._is_compiled = True
        self._dynamic_compile = dynamic
        self._curr_pre = self.update_pre()
        self._curr_post = self.update_post()
        
#        self._rates_function = types.MethodType( self._rates_function, self 
        return func_str
    
    def _compile_timeEvents(self):
        
        events = sorted(self._time_events)
        s = "@nb.njit\n"
        s += "def time_event(n, st, pars):\n"
        
        for i, te in enumerate(sorted(events)):
            s += "\tif n == %d:\n" % te._id
            s += "\t\t" + te.action + "\n"
        s += "\treturn None\n"
        s += "self._time_events_f = time_event"
        print(s)
        s = self._sub_vars(s)
        print(s)
        exec(s)
        
    
    def _sub_vars(self, s, par_name="pars", place_name = "st", dynamic = True):
        for i, name in enumerate(self.params.keys()):
            if(dynamic):
                s = re.sub("\\b" + name + "\\b", "%s[%d]" % (par_name,i), s)
            else:
                s = re.sub("\\b" + name + "\\b", "%e" % self.params[name], s)
        for i, name in enumerate(self.init_state.keys()):
            s = re.sub("\\b" + name + "\\b", "%s[%d]" % (place_name,i+1), s)
        return s
    
    def update_pre(self, st=None, pre=None):
        if st is None:
            st = self._state
        if pre is None:
            pre = np.zeros(shape = self._pre.shape, dtype = np.int32)
            for i in range(pre.shape[0]):
                for j in range(pre.shape[1]):
                    pre[i,j] = self._pre[i,j] if type(self._pre[i,j]) is not str else 0
        pars = np.array(list(self.params.values()))                    
        self._update_pre(st, pars, pre)
        return pre
    
    def update_post(self, st=None, post=None):
        if st is None:
            st = self._state
        if post is None:
            post = np.zeros(shape = self._post.shape, dtype = np.int32)
            for i in range(post.shape[0]):
                for j in range(post.shape[1]):
                    post[i,j] = self._post[i,j] if type(self._post[i,j]) is not str else 0
        pars = np.array(list(self.params.values()))                    
        self._update_post(st, pars, post)
        return post
    
    def simulate(self, tr_count=1, ODE = False, ret_raw=False, max_steps = 1e9, verbose = True):
        if not self._is_compiled:
            self.compile_system(dynamic=True)
        cpu_time = time.time()
        self._state = list(self.init_state.values())
        self._state.insert(0,0.)
        self._state = np.array(self._state, dtype=np.float64)
        if(verbose and False):
            print("simulate " + self.param_str())
        results={}
        params = self.get_all_params()
        tt = self.raster
        pre = self.update_pre()
        post = self.update_post()
#        print("AAAA:\n", globals())
        self._constants = np.array(list(self.params.values()))
        dim = (len(tt),) + self._state.shape
        _last_results = []
        t_events = sorted(self._time_events.copy())
        t_events.append(None)
        
        for i in range(tr_count):
            t = time.time()
            self._constants = np.array(list(self.params.values()))
            state = np.copy(self._state)
            STATES = np.zeros(dim)
            steps = 0
            t_low = 0.
            for k, te in enumerate(t_events):
#                print("Event: ", te)
                if t_low >= tt[-1]:
                    break
                if te is None:
                    t_high = tt[-1]
                else:
                    t_high = te.t
                indx = np.where((tt>=t_low) * (tt <=t_high))[0]
                if(len(indx)>0):
                    il = indx[0]
                    ih = indx[-1]+1
                else:
                    il = 0
                    ih = il
                steps += compute_stochastic_evolution(state,
                                                      t_high,
                                                      self._constants,
                                                      STATES[il:ih],
                                                      tt[il:ih],
                                                      self._rates_function,
                                                      self._update_pre,
                                                      self._update_post,
                                                      pre, post,
                                                      nb.int64(max_steps))
                if te is not None:
                    state[0] = te.t
                    print(steps)
                    print("EVENT ",te._id)
                    self._time_events_f(te._id, state, self._constants)
                t_low = t_high
            _last_results.append(STATES)
            t = time.time() - t
            if verbose:
                print("Steps: ", steps)
                print("runtime: ", t)
        self._last_results = np.array(_last_results)
#        if ret_raw:
#            results["stochastic"] = sim_st
#        sim_st_raster = rasterize(sim_st, tt)
        results["stoch_rastr"] = STATES
        
        #determinisitic simulation
        if ODE or self.simulate_ODE:
            sol_deterministic = self._simulate_ODE()
            #add time column
            print(sol_deterministic.shape)
            results["ODE"] = np.hstack((tt.reshape(len(tt),1),sol_deterministic))
        self.results = results
        cpu_time = time.time() - cpu_time
        if verbose:
            print("runtime: total", cpu_time )
        return results
    
    def _simulate_ODE(self, start_t=0.):
        t_events = self._time_events.copy()
        t_events.append(None)
        t_low = start_t
        raster = self.raster
        self._constants = np.array(list(self.params.values()))
        results = []
        y_0 = np.array(list(self.init_state.values()), dtype="float64")
        ih_old = -1
        for k, te in enumerate(t_events):
            if t_low >= raster[-1]:
                break
            if te is None:
                t_high = raster[-1]
            else:
                t_high = te.t
            indx = np.where((raster>=t_low) * (raster <=t_high))[0]
            if(len(indx)>0):
                il = indx[0]
                ih = indx[-1]
            else:
                il = 0
                ih = il
            tt = raster[il:ih+1]
            tt = np.insert(tt, 0, t_low)
            tt = np.append(tt, t_high)
            print("Rasterlen: ", len(tt))
            res = odeint(get_ODE_delta,y_0,tt,args = (self,))
            print("Res Shape: ", res.shape)
            y_0 = np.insert(res[-1], 0, t_high)
            if te is not None:
                self._time_events_f(te._id, y_0, self._constants)
            y_0 = np.delete(y_0, 0)
            res = np.delete(res, [0,len(res)-1],0)
            if(ih_old == il):
                res = np.delete(res, 0,0)
            ih_old = ih
            if (len(res) > 0):
#                print("Append.... ", res.shape)
                results.append(res)
            t_low = t_high
        
#        res = results[0]
#        #clean similar connections
#        for r in results[1:]:
#            while(len(r) > 0 and (res[-1][0] == r[0][0])):
#                r = np.delete(r,0,0)
#                print("deleteing....")
#            if len(r)>0:
#                res = np.concatenate([res,r])
        results = np.concatenate(list(results))
        print("Before Reshape: ", results.shape)
        #remove/flatten first dimension
#        print(results)
#        results = results.reshape(-1, res.shape[-1])
        return results
#            return np.hstack((tt.reshape(len(tt),1),results))
            
    
   
    
    
    
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
   
    def get_res_by_expr(self, expr, t_bounds=(0,np.inf)):
        
        if re.search("\\b_pre\[", expr) is not None:
            return self.get_res_by_expr_2(expr, t_bounds)
        indx = np.where((self.raster >= t_bounds[0]) * (self.raster <= t_bounds[1]))[0]
        _st = self.get_result("stoch_rastr")[indx]
        expr = self._sub_vars(expr, par_name="_pars", place_name="_st")
        expr = "lambda t, _st, _pars: " + expr
        f = eval(expr)
        res = np.zeros(len(_st))
        for i, _s in enumerate(_st):
            res[i] = f(_s[0], _s, self._constants)
        return res
    
    def get_res_by_expr_2(self, expr, t_bounds=(0,np.inf)):
        
        indx = np.where((self.raster >= t_bounds[0]) * (self.raster <= t_bounds[1]))[0]
        _pre = self.update_pre()
        st = self.results["stoch_rastr"]
        _pars = self._constants
        _u_pre = self._update_pre
        f_str = "def _f(_st, _pars, _u_pre, _pre):\n"
        f_str += "\tt=_st[0]\n"
        f_str += "\t_u_pre(_st, _pars, _pre)\n"
        expr = self._sub_vars(expr, par_name="_pars", place_name="_st")
        f_str += "\treturn " + expr + "\n"
        f_str += "self._f_tmp = _f\n"
#        print("AAAAAAAAAAA\n",f_str)
        exec(f_str )
        _f = np.vectorize(self._f_tmp, otypes=[np.float], excluded=(1,2,3),
                          signature="(n)->()")
        res = _f(st[indx], _pars, _u_pre, _pre)
        return res
    
    
    def get_res_index(self, name):
        if name not in self.init_state.keys():
            return None
        return list(self.init_state.keys()).index(name) +1
   
    
    #aux for plotting
    def _get_indices_and_colors(self, products=[], cmap="gist_ncar"):
        indices = []
        if not hasattr(self, "colors"):
            cmap = cm.get_cmap(cmap)
            self.colors = cmap(np.linspace(0, 1, len(self.init_state)+1))
#            self.colors = cm.rainbow(np.linspace(0, 1, len(self.init_state)+1))
        if isinstance(products, str):
            return [self.get_res_index(products)], colors
        for name in products:
            index = self.get_res_index(name)
            if(index is not None):
                indices.append(index)
        return indices, self.colors
    
    def _get_color(self, name = None):
        if not hasattr(self, "colors"):
            self._get_indices_and_colors()
        return self.colors[self.get_res_index(name)-1]
    def _set_color(self, name, c):
        i = self.get_res_index(name) -1
        self.colors[i] = colors.to_rgba_array(c)
        
        
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
            products = [products]
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
                         color = color, lw = 0.5*line_width,drawstyle = 'steps-post')[0])
                
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
            fig, ax = plt.subplots(1, figsize=(10*scale,10*scale))
        if type(products) == str:
            products = [products]
        if len(products) == 0:
            products = list(self.init_state.keys())[0]
            
        indx = np.where((self.raster >= t_bounds[0]) * (self.raster <= t_bounds[1]))[0]
        tt = self.raster[indx]
        indices, colors = self._get_indices_and_colors(products)

        for index, col, p in zip(indices, colors, products):
            results = self._last_results[:,indx,index]
            mean = np.mean(results, axis = 0)
            ax.plot(tt, results.T, c=col, lw=0.2, alpha=0.5)
            ax.plot(tt, mean, c = col, lw=3, label = p, alpha=1)
        return ax
    
    
    
    def plot_par_var_1d(self, par = "s1", vals = [1,2,3,4,5], label = None,
                        ax=None, plot_args = dict(), func=None, **func_pars):
        res = []
        for v in vals:
            self.set_param(par, v)
            self.simulate()
            res.append(func(**func_pars))
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(vals, res, label = label, **plot_args)
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
#    print(reacts, rates)
    dx = reacts.transpose().dot(rates)
    return dx

@nb.njit#(nb.f8(nb.f8[:,:], nb.f8[:], nb.f4, nb.f8[:](nb.f8[:],), nb.i4))
def compute_stochastic_evolution(state, t_max, constants, STATES, time_steps, rate_func,
                                 pre_upd_f, post_upd_f , pre, post, max_steps):
    tt = state[0]
    steps = nb.int64(0)
    i = 0
    length = len(time_steps)
    rates_array = np.zeros(len(pre))
    rr_count = int(1e5)
    j=0
    while steps < max_steps:
        if j % rr_count ==0:
            rr = np.random.uniform(0,1,size=2*rr_count)
            j=0
        # sample two random numbers uniformly between 0 and 1
#        rr = np.random.uniform(0,1,2)
        r1 = rr[j*2]
        r2 = rr[j*2+1]
        j+=1
#        r1, r2 = np.random.uniform(0,1,2)
#        print(reactions)
        pre_upd_f(state, constants, pre)
        post_upd_f(state, constants, post)
        a_s = rate_func(state, constants, rates_array, pre)
#        print(a_s)
        a_0 = a_s.sum()
        # time step: Meine Version
        #print(a_0)
        if a_0 == 0:
#            tt = time_steps[-1]
            tt = t_max
        else:
            tt = tt - np.log(1. - r1)/a_0
        
        state[0] = tt
        while(tt >= time_steps[i] and i < length):
            STATES[i,:] = state
            STATES[i,0] = time_steps[i]
            i+=1
        if(tt >= t_max):
            break
        # find the next reaction
        prop = r2 * a_0
        cum_a_s = np.cumsum(a_s)
        ind = np.where(prop <= cum_a_s)[0][0]
        # update the systems state
        state[1:] +=post[ind] - pre[ind]
#        transition_funcs[ind](state, constants) # does not work :(
        
        steps+=1

#    print("Steps needed: ", steps)
    return steps
# define parameters  
