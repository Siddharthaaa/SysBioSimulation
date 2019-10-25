# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:45:28 2019

@author: imb30
"""

import os

from tkinter import Tk # copy to clipboard function
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk
from tkinter.colorchooser import askcolor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import gridspec
from scipy.optimize import fsolve
from scipy.integrate import odeint
#other ode solver. IMPORTANT
#from scikits.odes import ode
import pickle
import re
import numba as nb
import numba.typed
from numba.typed import List
from sklearn.cluster import KMeans, MeanShift, k_means
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64
import random
drawPetriNets = True
if(drawPetriNets):
    import snakes
    import snakes.plugins
    snakes.plugins.load("gv","snakes.nets","nets")
#    snakes.plugins.load('clusters', 'nets', 'snk')
    import nets as pns


# inline displaying
#%matplotlib inline
# settings for plots
#matplotlib.rcParams['axes.labelsize'] = 16
#matplotlib.rcParams['xtick.labelsize'] = 16
#matplotlib.rcParams['ytick.labelsize'] = 16
#matplotlib.rcParams['legend.fontsize'] = 14
#matplotlib.rcParams['font.family'] = ['sans-serif']
#line_width = 2
#color_pre = 'dodgerblue'
#color_Incl = 'darkorange'
#color_Skip = "green"
#color_phase_space = 'dimgrey'
#color_initial_state = 'crimson'
#color_steady_state = 'saddlebrown'

class empty(object):
    def __init__(self):
        pass

class TimeEvent(object):
    
    _count = 0    
    def __init__(self, t, a, name = None):
        if (name is None):
            name = "TimeEvent" + str(TimeEvent._count + 1)
            TimeEvent._count+=1
        self.name = name
        self.t = t
        self.action = a
    def __lt__(self, te):
        return self.t < te.t
    def __le__(self, te):
        return self.t <= te.t
    def __gt__(self, te):
        return self.t > te.t
    def __ge__(self, te):
        return self.t >= te.t
    def __str__(self):
        return self.name + ":" + " at " + str(self.t) + "\nAction: " + self.action 
    def __repr__(self):
        return  self.__str__()

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
        self._time_events.append(te)
        self._time_events = sorted(self._time_events)
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
        
    
    def draw_pn(self, filename=None, rates=False, rotation = False,
                engine=('neato', 'dot', 'circo', 'twopi', 'fdp'),
                draw_neutral_arcs = True,
                draw_inhibition_arcs = True,
                draw_flush_arcs = True,
                **kwargs):
        if type(engine) is str:
            engine = (engine,)
        self.compile_system()
        #https://www.ibisc.univ-evry.fr/~fpommereau/SNAKES/API/plugins/gv.html
        if filename is None:
            filename = self.name + ".png"
            filename = os.path.join("pn_images", filename)
            path = os.path.dirname(filename)
            if(not os.path.exists(path)):
                os.makedirs(path)
            
        pn = pns.PetriNet(self.name)
        for i, (p, v) in enumerate(self.init_state.items()):
            cluster = self._clusters[p] if p in self._clusters else ()
#            pn.add_place(pns.Place(p,v), cluster=cluster)
            pn.add_place(pns.Place(p,v), cluster=cluster)
        
        for i, (tr_name, pre, post) in enumerate(zip(self._transitions.keys(), self._pre, self._post)):
            name = tr_name
            tr = self._transitions[tr_name]
            cluster = self._clusters[name] if name in self._clusters else ()
            if(rates):
                pn.add_transition(pns.Transition(name, pns.Expression(tr["rate"])),
                                                 cluster = cluster)
            else:
                pn.add_transition(pns.Transition(name), cluster=cluster)
                
            #creating arcs
            for pr, pst, subs in zip(pre, post, list(self.init_state)):
                if(pr == pst and pr != 0): #neutral 
                    if(draw_neutral_arcs):
                        v = pns.Value(pr)
                        v._role = "neutral"
                        pn.add_input(subs, name, v)
                else:
                    # ugly stuff starts :(
                    if(pr == "2*" + subs ): #inhibition
                        if(draw_inhibition_arcs):
                            v = pns.Value("<inhibits>")
                            v._role = "inhibition"
                            pn.add_input(subs, name, v)
                    elif(pr == subs): #flush
                        if(draw_flush_arcs):
                            flush = pns.Flush("0")
                            flush._role = "flush"
                            pn.add_output(subs, name, flush)
                    elif(pr != 0):
                        pn.add_input(subs, name, pns.Value(pr))
                    if(pst != 0):
                        pn.add_output(subs, name, pns.Value(pst))
                    
         
        #documentation of attr                
        #http://www.graphviz.org/doc/info/attrs.html        
        def draw_place (place, attr) :
#            print(attr)
            attr['label'] = place.name
            attr['color'] = colors.to_hex(self._get_color(place.name))
        def draw_transition (trans, attr) :
#            print(attr)
            if str(trans.guard) == 'True' :
                attr['label'] = trans.name
            else :
                attr['label'] = '%s\n%s' % (trans.name, trans.guard)
        def draw_arc(arc, attr):
            #arrow styles
            #http://www.graphviz.org/doc/info/attrs.html#k:arrowType
            if(hasattr(arc, "_role")):
                if(arc._role == "inhibition"):
                    attr["arrowhead"] = "tee"
#                    attr["arrowhead"] = "odot"
                    attr["style"] = "bold"
                    attr["label"] = "inhibition"
                    pass
                if(arc._role == "flush"):
#                    attr["arrowhead"] = "odot"
                    attr["label"] = "0"
                    attr["arrowhead"] = "empty"
                    attr["style"] = "dashed"
                    pass
                if(arc._role == "neutral"):
                    attr["dir"] = "both"
#                    attr["arrowhead"] = "box"
#                    attr["arrowtail"] = "box"
                    attr["style"] = "dotted"
                    attr["arrowhead"] = "ediamond"
                    attr["arrowtail"] = "ediamond"
                    pass
        def draw_graph(g, attr):
#            print("AAAAAAAAA", g)
#            print(attr)
            attr["rotate"] = 0
            attr["style"] = "invis"
#            attr["style"] = "dashed"
            attr["ratio"] = 2
            attr["rankdir"] = "LR"
#            attr["bgcolor"] = "#ff0000"
#            attr["label"] = "KOMM SCHON"
        if(rotation):
            pn.transpose()
        for e in engine:
            f_name = re.sub("(\.\w+)$", "_"+ e + "\\1", filename)
            pn.draw(f_name, engine = e, debug=True,
                place_attr=draw_place,
                trans_attr=draw_transition ,
                arc_attr = draw_arc,
                graph_attr = draw_graph,
                cluster_attr = draw_graph,
                **kwargs)
        return pn
    
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
            s += "\tif n == %d:\n" % i
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
        state = np.copy(self._state)
        dim = (len(tt),) + state.shape
        _last_results = []
        t_events = self._time_events.copy()
        t_events.append(None)
        t_low = 0.
        for i in range(tr_count):
            t = time.time()
            STATES = np.zeros(dim)
            steps = 0
            for k, te in enumerate(t_events):
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
#                    print(steps)
#                    print("EVENT ",k)
                    self._time_events_f(k, state, self._constants)
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
        t_high_old = np.inf
        for k, te in enumerate(t_events):
            if te is None:
                t_high = raster[-1]
            else:
                t_high = te.t
            indx = np.where((raster>=t_low) * (raster <=t_high))[0]
            if(len(indx)>0):
                il = indx[0]
                ih = indx[-1]+1
            else:
                il = 0
                ih = il
            tt = raster[il:ih]
            if t_low not in tt:    
                tt = np.insert(tt, 0, t_low)
            if t_high not in tt:
                tt = np.append(tt, t_high)
            print("Rasterlen: ", len(tt))
            res = odeint(get_ODE_delta,y_0,tt,args = (self,))
            print("Res Shape: ", res.shape)
            y_0 = np.insert(res[-1], 0, t_high)
            if te is not None:
                self._time_events_f(k, y_0, self._constants)
            y_0 = np.delete(y_0, 0)
            if(t_high_old == t_low):
                res = np.delete(res, 0,0)
            t_high_old = t_high
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
            
    
    def simulate_cuda(self, params = {"s1": [1,2,3,4,5,6,7,8,9],
                                      "s2": [0.4,0.5,0.7,1,2,3]},
                        max_steps = 1000, fallback=True):
        #creating param array
        
        
        self.compile_system(dynamic=True)
#        dim1 = len(list(params.values())[0])
        
        dim = ()
        for k,v in params.items():
            dim += (len(v), )
#            for i, par_i in enumerate(v):
        
        out_dim = dim +  (len(self.raster), len(self.init_state)+1 )
        
        self.cuda_params_dict = params
        
        if(not cuda.is_available()):
            print ("CUDA not available")
            if(fallback):
                print("Fallback is on. Exection an CPU...")
                
            return None
        
        gpus = cuda.gpus
        dev = cuda.gpus.current
#        con = cuda.cudadrv.driver.Context(dev, None)
        con = cuda.current_context()
        con.reset()
        mem = con.get_memory_info()

        mem_estim = np.prod(np.array(out_dim, dtype = np.int64)) * 8
        mem_ratio = mem_estim/mem.free
        print("Mem ratio: ", mem_ratio)
        print("Mem free: ", mem.free)
        print("Mem estimate: ", mem_estim)
#        print(dim)
#        print(out_dim)
        if( mem_ratio> 0.8 or mem.free < 1e8):
            print("Memory need: ", mem_estim, "\nMemory available: ", mem.free)
            print("Ratio ", mem_ratio, " exceeds ", 0.8 )
            print("Abort calculation")
            return None
        
        threads_per_block = np.prod(dim)
        blocks = 1
        all_params = np.zeros(dim + (len(self.params),), dtype = np.float64)
#        print(all_params)
        out = np.zeros((dim +  (len(self.raster), len(self.init_state)+1 )), dtype=np.float64)
        indx = np.zeros(len(dim), dtype=int)
        keys, values = list(params.keys()), list(params.values())
        max_ind = np.array(dim)
        deep = 0
        while indx[0] < max_ind[0] and deep >= 0:
            while indx[deep] < max_ind[deep]:
                if(deep < len(dim)-1):
                    deep += 1
                else:
#                    print(indx)
                    all_params[tuple(indx)]= np.fromiter(self.params.values(), dtype= float)
                    out[tuple(indx)][0][1:] = np.array(list(self.init_state.values()))
                    for i, k in enumerate(keys):
                        ind = list(self.params).index(k)
#                        print("ind: ", ind)
#                        print("all_par: ", all_params[indx])
                        all_params[tuple(indx)][ind] = params[k][indx[i]]
#                        print("all_par: ", all_params[indx])
                    indx[deep] += 1
            indx[deep] = 0
            deep -= 1
            indx[deep] += 1
#        return all_params
#        for k,v in params.items():
#            ind = list(self.params).index(k)
#            for i, par_i in enumerate(v):
#                all_params[i]= np.fromiter(self.params.values(), dtype= float)
#                
#                all_params[i][ind] = par_i
#        
#        gpu_test_func = cuda.jit(device=True)(lambda x: x*x)
        reactions = np.array(self.get_reacts(), dtype=np.int32)
        d_reactions = cuda.to_device(reactions)
        d_out = cuda.to_device(out)
        self.cuda_last_params = all_params
        d_all_params = cuda.to_device(all_params)
#        print(all_params)
        
        pre = self.update_pre()
        post = self.update_post()
        pre_buff = np.tile(pre, (dim[0],dim[1],1,1))
        post_buff = np.tile(post, (dim[0],dim[1],1,1))
        d_pre_buff = cuda.to_device(pre_buff)
        d_post_buff = cuda.to_device(post_buff)
        raster = np.array(self.raster, np.float32)
#        rng_states = create_xoroshiro128p_states(blocks * threads_per_block * 2, seed=1)
        rng_states = create_xoroshiro128p_states(2048, seed=1)
        rates_buff = np.zeros(dim + (len(self.get_reacts()),))
        d_rates_buff = cuda.to_device(rates_buff)
        progress_indx = np.zeros(dim, dtype=np.int32)
        d_progr_indx = cuda.to_device(progress_indx)
        log_arr = np.zeros(dim)
        d_log_arr = cuda.to_device(log_arr) 
        if True:
#        if not hasattr(self, "compute_stochastic_evolution_cuda"):
            gpu_rates_func = cuda.jit(device=True)(self._rates_function)
            self._gpu_rates_f = gpu_rates_func
            
            gpu_update_pre = cuda.jit(device=True)(self._update_pre)
            gpu_update_post = cuda.jit(device=True)(self._update_post)
            
            @cuda.jit
            def compute_stochastic_evolution_cuda(STATES, reacs, rates_b, constants_all, pre_all, post_all,
                                                  time_steps, max_steps, rng_st, progr_i, log):

                th_nr = cuda.grid(2)
                #TODO 
                x, y = th_nr[0], th_nr[1]
                thid = cuda.blockDim.x * y + x
                
                STATES = STATES[x,y]
                constants_all = constants_all[x,y]
                pre_all = pre_all[x,y]
                post_all = post_all[x,y]
                rates_b = rates_b[x,y]
               
                steps = nb.int32(0)
                t_ind = progr_i[x,y]
#                t_log[x,y] = t_ind
                length = len(time_steps)
                tt=time_steps[t_ind]
                
                
                
                while steps < max_steps and t_ind < length:

                    r1 = xoroshiro128p_uniform_float32(rng_st, thid*2)
                    r2 = xoroshiro128p_uniform_float32(rng_st, thid*2+1)
                    
#                    pre_upd_f(state, constants, pre)
#                    post_upd_f(state, constants, post)
#                    a_s = rate_func(state, constants, rates_array, pre)
                    
                    gpu_update_pre(STATES[t_ind], constants_all, pre_all)
                    gpu_update_post(STATES[t_ind], constants_all, post_all)
                    
                    gpu_rates_func(STATES[t_ind], constants_all, rates_b, pre_all)
                    a_0 =0
                    for i in range(len(rates_b)):
                        a_0 += rates_b[i]
                    
                    if a_0 == 0:
                        break
                    tt = tt - math.log(1. - r1)/a_0
                    STATES[t_ind][0] = tt
                    while(tt >= time_steps[t_ind] and t_ind < length):
                        if(t_ind < length-1):
                            for k in range(len(STATES[t_ind])):
                                STATES[t_ind+1][k] = STATES[t_ind][k]
                                
    #                        STATE[t_ind+1] = STATE[t_ind]
                        STATES[t_ind,0] = time_steps[t_ind]
                        
                        t_ind+=1
                    # find the next reaction
                    prop = r2 * a_0
                    a_sum = 0.
                    ind = 0
                    # DOTO: BUG ASSUMED
                    for i in range(len(rates_b)):
                        a_sum += rates_b[i]
#                        if prop >= a_sum and prop < rates_b[i]+a_sum:
                        if a_sum >= prop:
                            ind = i
                            break
                        
                    # update the systems state
                    for j, (pr,po) in enumerate(zip(pre_all[ind], post_all[ind])):
                        STATES[t_ind][j+1] += po-pr
                    steps+=1
                    progr_i[x,y] = t_ind
            self.compute_stochastic_evolution_cuda = compute_stochastic_evolution_cuda
            
        i = 0
        self.cuda_out = out
#        d_progr_indx = cuda.to_device(progress_indx)
        while(np.any(progress_indx < len(self.raster)-1)):
#            print("loop ", i )
            if (i % 100 == 0): print("loop: ", i , "\n", progress_indx)
#            d_progr_indx = cuda.to_device(progress_indx)
            self.compute_stochastic_evolution_cuda[blocks, dim](d_out,
                                             d_reactions, d_rates_buff,
                                             d_all_params,
                                             d_pre_buff, d_post_buff,
                                             raster, max_steps,
                                             rng_states, d_progr_indx, d_log_arr )

            i+=1
#            d_out.copy_to_host(self.cuda_out)
            d_progr_indx.copy_to_host(progress_indx)
#            d_log_arr.copy_to_host(log_arr)
#            self.cuda_log = log_arr
            
        d_rates_buff.to_host()
        d_out.to_host()
        self.cuda_out = out
        return out
    
    def _create_params_array(self, params = {"s1": [8,9,10], "s2": [0.4,0.5,0.7]}):
        dim = ()
        for k,v in params.items():
            dim += (len(v), )
        all_params = np.zeros(dim + (len(self.params),), dtype = np.float32)
#        print(all_params)
        indx = np.zeros(len(dim), dtype=int)
        keys, values = list(params.keys()), list(params.values())
        max_ind = np.array(dim)
        deep = 0
        while indx[0] < max_ind[0] and deep >= 0:
            while indx[deep] < max_ind[deep]:
                if(deep < len(dim)-1):
                    deep += 1
                else:
#                    print(indx)
                    all_params[tuple(indx)]= np.fromiter(self.params.values(), dtype= float)
                    for i, k in enumerate(keys):
                        ind = list(self.params).index(k)
#                        print("ind: ", ind)
#                        print("all_par: ", all_params[indx])
                        all_params[tuple(indx)][ind] = params[k][indx[i]]
#                        print("all_par: ", all_params[indx])
                    indx[deep] += 1
            indx[deep] = 0
            deep -= 1
            indx[deep] += 1
        return all_params
    
    def _create_out_array(self, shape = (5,2)):
        
        out = out = np.zeros((shape +  (len(self.raster), len(self.init_state)+1 )), dtype=np.float32)
        
        deep = 0
        stack = []
        i = 0
        pointer = out
        while deep >= 0 and i < len(pointer):
            while len(pointer.shape) > 2:
                stack.append((i, pointer))
                pointer = pointer[i]
                i = 0
            pointer[0][1:] = np.array(list(self.init_state.values()))
            i, pointer = stack.pop()
            i += 1
            if(i > len(pointer)):
                deep -= 1
                i, pointer = stack.pop()
                #//TODO OOOOOO
        
        return out
    
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
    
    def get_psi_cv(self, **kwargs):
        psi = self.compute_psi(**kwargs)[1]
        sd, mean = np.std(psi), np.mean(psi)
        return sd/mean
    def get_psi_mean(self, **kwargs):
        self.compute_psi(**kwargs)
        return np.mean(self.results["PSI"][1])
    def get_res_index(self, name):
        if name not in self.init_state.keys():
            return None
        return list(self.init_state.keys()).index(name) +1
    def compute_psi(self, products = ["Incl", "Skip"], solution="stoch_rastr",
                    ignore_extremes = False, ignore_fraction=0.1, recognize_threshold = 1,
                    exact_sum = None, sim_rnaseq = None):
        
#        print("compute psi...")
        sim_st_raster = self.results[solution]
        start_ind = int(len(sim_st_raster)*ignore_fraction)
        incl_counts = np.array(self.get_res_col(products[0]), dtype=np.int32)
        skip_counts = np.array(self.get_res_col(products[1]), dtype=np.int32)
        
        if(sim_rnaseq is not None):
            incl_counts = sp.stats.binom.rvs(incl_counts, sim_rnaseq)
            skip_counts = sp.stats.binom.rvs(skip_counts, sim_rnaseq)
        
        indices =  np.array(np.where(incl_counts + skip_counts >= recognize_threshold))
        if ignore_extremes:
            indices_extr = [np.where((incl_counts != 0) * (skip_counts != 0))]
            indices = np.intersect1d(indices, indices_extr)
        if exact_sum is not None:
            indices_extr = [np.where(incl_counts + skip_counts == exact_sum)]
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
                lines.append(ax.plot(stoch_res[indx,0],stoch_res[indx,index], label = name +"(stoch)",
                         color = color, lw = 0.5*line_width,drawstyle = 'steps')[0])
                
            mean = np.mean(stoch_res[int(len(stoch_res)/3):,index])
            #plot mean of stoch sim
            if plot_mean:
                ax.plot([tt[0],tt[-1]], [mean,mean], "--", color=color, lw=line_width)
            if "ODE" in res and ode_res is not None:
                ax.plot(tt,ode_res[indx,index],"--", color = color, lw = 1.5*line_width, label = name + "(ODE)")
        
        #plot expressions
        for i, e in enumerate(exprs):
            res = self.get_res_by_expr(e, t_bounds = t_bounds)
            ax.plot(tt,res, "-.", color ="C"+str(i), lw=0.7*line_width, label =e)
        
        #ax.yaxis.set_label_coords(-0.28,0.25)
        if len(products2)>0:
            if type(products2) is str:
                products2 = [products2]
            ax_2 = ax.twinx()
            ax_2 = self.plot_course(ax = ax_2, products=products2, t_bounds = t_bounds, clear=True)
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
        return ax
    
    def plot_series(self, ax=None, products=[], t_bounds=(0, np.inf), scale=1):
        if ax == None:
            fig, ax = plt.subplots(1, figsize=(10*scale,10*scale))
        if type(products) == str:
            products = [products]
        if len(products) == 0:
            products = list(self.init_state.keys()[0])
            
        indx = np.where((self.raster >= t_bounds[0]) * (self.raster <= t_bounds[1]))[0]
        tt = self.raster[indx]
        indices, colors = self._get_indices_and_colors(products)

        for index, col, p in zip(indices, colors, products):
            results = self._last_results[:,indx,index]
            mean = np.mean(results, axis = 0)
            ax.plot(tt, results.T, c=col, lw=0.2, alpha=0.5)
            ax.plot(tt, mean, c = col, lw=3, label = p, alpha=1)
        return ax
    
    def plot_cuda(self, **kwargs):
# plots a 2D grid of simulations
#        fig = plt.figure()
        x,y = tuple(self.cuda_last_params.shape[:2])
        fig, ax = plt.subplots(x, y, sharex =True, sharey = True)
        cuda_res = self.cuda_out
        res_tmp = self.results["stoch_rastr"]
        for xx in range(x):
            for yy in range(y):
                self.results["stoch_rastr"]=cuda_res[xx,yy]
                self.plot_course(ax = ax[xx,yy], res=["stoch"], **kwargs)
                ax[xx,yy].set_ylabel("")
                ax[xx,yy].set_xlabel("")
                ax[xx,yy].set_title("")
                ax[xx,yy].get_legend().remove()
                
                
        self.results["stoch_rastr"] = res_tmp
    
    def plot_par_var_1d(self, par = "s1", vals = [1,2,3,4,5], label = None,
                        ax=None, func=None, **func_pars):
        res = []
        for v in vals:
            self.set_param(par, v)
            self.simulate()
            res.append(func(**func_pars))
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(vals, res, label = label)
        ax.set_ylabel(func.__func__.__name__ +   str(func_pars))
        ax.set_xlabel(par)
        
        return ax
    
    def plot_par_var_2d(self, pars = {"s1":[1,2,3], "s2": [1,2,4]},ax = None, func=None, **func_pars):
        
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
        
        heatmap(np.array(res), params[0], params[1], ax, cbarlabel= func.__func__.__name__ +   str(func_pars) )
        ax.set_xlabel(", ".join(names[1]))
        ax.set_ylabel(", ".join(names[0]))
        
        results = empty()
        results.ax = ax
        results.values = res
        results.sim_res = sim_res
        return results
    
class SimInterface(tk.Frame):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.initUI(sim)
    
    def initUI(self, sim):
        self.sim = sim
        self.master.title(sim.name)
        self.pack(fill=tk.BOTH, expand=True)
                
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)
        
        f_settings = self._create_settings_f(self, sim)
        f_settings.pack(side=tk.TOP, fill=tk.X)
        f_places = self._create_graph_selection(self, sim)
        f_places.pack(fill=tk.X, side=tk.LEFT)
        
        f_plot = tk.Frame(self)
        f_plot.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        par_sbar = tk.Scrollbar(self, orient="vertical" )
        scroll_geometry = (0, 0, 1000, 1000)
        f_params_container = tk.Canvas(self, scrollregion=scroll_geometry,
                                       yscrollcommand=par_sbar.set)
        
        f_params = tk.Frame(f_params_container)
        f_params.pack(side=tk.BOTTOM, fill = tk.X, expand=True)
        
        par_sbar.pack(side=tk.RIGHT, fill=tk.Y)
        f_params_container.configure(yscrollcommand=par_sbar.set)
        f_params_container.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        par_sbar.config(command=f_params_container.yview)
        
#        def key_pressed(e):
#            print("key pressed: ", e.char)
#            if(e.char == "\n"): self.update(True)
#        f_params.bind('<Return>', lambda e: key_pressed(e) ) 
        
        self._show_sp = [[],[]]
        self._show_sp
        i = 0
        self.par_entries = {}
       
        for k, v in sim.params.items():
#            row = tk.Frame(f_params)
#            row.pack(side=tk.TOP, fill=tk.X, padx = 1, pady=1)
            label = tk.Label(f_params, text=k)
            label.grid(row=i, column=0)
            entr = tk.Entry(f_params)
            entr.insert(0,v)
            entr.grid(row=i, column=1)
            entr.bind('<Return>', lambda e: self.update(True) ) 
            self.par_entries[k]= entr
            i +=1
            
       
            
        fig = plt.Figure(figsize=(5,5))
        self._fig = fig
        canvas = FigureCanvasTkAgg(fig, f_plot)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self._canvas = canvas
        self._plot_sim()
        
        f_control = tk.Frame(self)
        f_control.pack(side =tk.BOTTOM, fill =tk.X)
        b_update = tk.Button(f_control, text="Update",
                             command = lambda: self.update(True))
        b_update.pack(side=tk.RIGHT)
    def _create_settings_f(self, master,sim):
        self._setting_e ={}
        #common settings frame
        f_settings = tk.Frame(self)
        l_runtime  = tk.Label(f_settings,text = "runtime:" )
        l_runtime.grid(row=0, column=0)
        l_raster = tk.Label(f_settings, text = "raster:")
        l_raster.grid(row=1, column=0)
        
        e_runtime = tk.Entry(f_settings)
        e_runtime.insert(0,str(sim.runtime))
        e_runtime.bind('<Return>', lambda e: self.update(True) ) 
        e_runtime.grid(row=0, column=1)
        self._setting_e["runtime"] = e_runtime
        
        e_raster = tk.Entry(f_settings)
        e_raster.insert(0,str(sim.raster_count))
        e_raster.bind('<Return>', lambda e: self.update(True)) 
        e_raster.grid(row=1, column=1)
        self._setting_e["raster_count"] = e_raster
        return f_settings
    
    def _create_graph_selection(self, master, sim):
        self._spezies_checkb = {}
        self._spezies_checkb2 = {}
        self._spezies_col_b = {}
        self.pl_entries = {}
        frame = tk.Frame(master)
        for i, (k, v) in enumerate(sim.init_state.items()):
            c = sim._get_color(k)
            checked = tk.IntVar()
            check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                       command= lambda : self.update(False))
#            check_box.select()
            check_box.grid(row=i, column=0)
            self._spezies_checkb[k] = checked
            checked = tk.IntVar()
            check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                       command= lambda : self.update(False))
#            check_box.select()
            check_box.grid(row=i, column=1)
            self._spezies_checkb2[k] = checked
            label = tk.Label(frame, text=k)
            label.grid(row=i, column =2)
            entr = tk.Entry(frame, width=4)
            entr.insert(0,v)
            entr.grid(row=i, column=3)
            self.pl_entries[k]= entr
            b_col = tk.Button(frame, height =1, width=1, bg = colors.to_hex(c),
                              command = lambda key=k: self._new_color(key))
            b_col.grid(row=i, column=3)
            self._spezies_col_b[k] = b_col
            
        #create transition(rate) selection
        i+=1
        self._transtion_checks = []
        checked = tk.IntVar()
        check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                   command= lambda : self.update(False))
        check_box.grid(row=i, column=0)
        self._transtion_checks.append(checked)
        checked = tk.IntVar()
        check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                   command= lambda : self.update(False))
        check_box.grid(row=i, column=1)
        self._transtion_checks.append(checked)
        
        transition_cb = ttk.Combobox(frame, values=list(sim._transitions))
        transition_cb.current(0)
        transition_cb.bind("<<ComboboxSelected>>", lambda e: self.update(False))
        transition_cb.grid(row=i, column=2)
        self._tr_cb = transition_cb
        
        #free expression Entry
        i+=1
        self._expr_checks = []
        checked = tk.IntVar()
        check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                   command= lambda : self.update(False))
        check_box.grid(row=i, column=0)
        self._expr_checks.append(checked)
        checked = tk.IntVar()
        check_box = tk.Checkbutton(frame, text="", variable = checked, #command = self.update)
                                   command= lambda : self.update(False))
        check_box.grid(row=i, column=1)
        self._expr_checks.append(checked)
        
        expr_e = tk.Entry(frame, width=20)
        expr_e.bind("<Return>", lambda e: self.update(False))
        expr_e.grid(row=i, column=2)
        self._expr_e = expr_e
            
        return frame
            
    def _new_color(self, name):
        b_c = self._spezies_col_b[name]
        col_new = askcolor(b_c.cget("bg"))[1]
        if col_new is not None:
            b_c.configure(bg=col_new)
            self.sim._set_color(name, col_new)
        self.update(False)
    
    def _plot_sim(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
#        ax.clear()
        print(self._show_sp)
        self.sim.plot_course(ax = ax, products = self._show_sp[0], products2=self._show_sp[1])
        ax.legend([])
        ax.set_ylabel("#", rotation = 0)
        self._canvas.draw()
    
    def update(self, sim=False):
        self.fetch_places()
        if(sim):
            self.fetch_pars()
            self.fetch_settings()
            self.sim.simulate()
        print("update..")
        self._plot_sim()
    def fetch_settings(self):
        sim = self.sim
        for k, v in self._setting_e.items():
            sim.__dict__[k] = eval(v.get())
        sim.set_raster()
        
    def fetch_places(self):
        sim = self.sim
        self._show_sp = []
        show_sp = []
        for k, checked in self._spezies_checkb.items():
            if checked.get():
                show_sp.append(k)
                
        show_sp2 = []
        for k, checked in self._spezies_checkb2.items():
            if checked.get():
                show_sp2.append(k)
                
        self._show_sp=[show_sp, show_sp2]
        
        for i, tr_cb in enumerate(self._transtion_checks):
            if(tr_cb.get()):
                self._show_sp[i].append(sim._transitions[self._tr_cb.get()]["rate_ex"])
        for i, expr_cb in enumerate(self._expr_checks):
            if(expr_cb.get()):
                self._show_sp[i].append(self._expr_e.get())
        
    def fetch_pars(self):
        sim = self.sim
        for k, e in self.par_entries.items():
            v = eval(e.get())
            sim.set_param(k,v)
        
            
        

        
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
    x = x[~np.isnan(x)]
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
#        for blabal in range(1000):
        pre_upd_f(state, constants, pre)
        post_upd_f(state, constants, post)
        a_s = rate_func(state, constants, rates_array, pre)
#        print(a_s)
        a_0 = a_s.sum()
        # time step: Meine Version
        #print(a_0)
        if a_0 == 0:
            tt = time_steps[-1]
        else:
            tt = tt - np.log(1. - r1)/a_0
        
        state[0] = tt
        while(tt >= time_steps[i] and i < length):
            STATES[i,:] = state
            STATES[i,0] = time_steps[i]
            i+=1
        if(tt > t_max):
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
        
def simulate(sim, ODE = True, ret_raw=False, max_steps = 1e9):
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

def get_exmpl_sim(name = ("basic", "LotkaVolterra", "hill_fb")):
    s = None
    if(type(name) != str):
        name = "basic"
    if(name == "basic"):
        s1= s2 = s3 = d1 = d2 = d3 = d0 = 1
        d1 = 0.1
        d2 = 0.1
        k_syn = 100
        
        name="Basic"
        s = SimParam(name,100, 1001,
                     params = {"v_syn": k_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
                     init_state = {"pre_RNA": 1, "Incl": 1, "Skip":1, "ret": 1})
        
        s.simulate_ODE = True
        
        s.add_reaction("v_syn", {"pre_RNA":1}, "Transcription" )
        s.add_reaction("d0*pre_RNA", {"pre_RNA":-1}, "mRNA degr." )
        s.add_reaction("s1*pre_RNA", {"pre_RNA":-1, "Incl":1}, "Inclusion")
        s.add_reaction("d1*Incl", {"Incl": -1} , "Incl. degr." )
        s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1}, "Skipping")
        s.add_reaction("d2*Skip", {"Skip":-1}, "Skip. degr."  )
        s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1}, "Retention" )
        s.add_reaction("d3*ret",  {"ret": -1}, "ret. degr" )
    elif(name == "test"):
        params = {"p1":1, "p2":2, "p3":3, "degr":100}
        s = SimParam(name, 100,1001, params,
                     init_state = {"A1": 10, "A2": 15, "A3": None})
        s.simulate_ODE = False
        s.add_reaction("p1*A1", {"A1":-1, "A3":2})
        s.add_reaction("p2*A2", {"A1":1, "A3":1})
        s.add_reaction("p3*A3", {"A3":-1, "A1":"A3*p2"})
        s.add_reaction("degr*A1", {"A1":-1 })
        s.add_reaction("degr*A3", {"A3":"-2*A1" })
        s.add_reaction("degr*A2", {"A2":-1 })
        
    elif(name == "hill_fb"):
                
        k_on=0
        k_off=0.00
        v_syn=10.000
        s1=0.500
        Ka1=70
        n1=6.000
        s2=1.000
        s3=0.100
        d0=0.100
        d1=0.100
        d2=0.100
        d3=0.500
        s1_t=5
        name="Hill_Feedback"
        s = SimParam(name, 400, 1001,
                     params = {"k_on": k_on, "k_off": k_off, "v_syn": v_syn, "s1": s1, "Ka1":Ka1, "n1":n1,
                               "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3, "s1_t":s1_t},
                     init_state = {"Pr_on": 1, "Pr_off": 0, "pre_RNA": 0,
                                   "Incl": 10, "Skip": 0, "ret": 0})
        
        s.simulate_ODE = True
        
        s.add_reaction("k_on*Pr_off", {"Pr_on":1, "Pr_off":-1}, "Prom. act.")
        s.add_reaction("k_off*Pr_on", {"Pr_off":1, "Pr_on":-1}, "Prom. deact.")
        s.add_reaction("v_syn*Pr_on", {"pre_RNA":1, "Pr_on":[-1,1]}, "Transcription")
        s.add_reaction("d0*pre_RNA", {"pre_RNA":-1}, "mRNA degr." )
        s.add_reaction("s1*pre_RNA + pre_RNA* s1_t * (1/(1 + (Ka1/Incl)**n1) if Incl > 0 else 0)",
                       {"pre_RNA":-1, "Incl":1}, "Inclusion")
        s.add_reaction("d1*Incl", {"Incl": -1}, "Incl. degr"  )
        s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1}, "Skipping")
        #            s.add_reaction("s2*pre_RNA + 5* (Skip > 0)* 1/(1+(Ka2/Skip)**n2)" ,  {"pre_RNA":-1, "Skip":1})
        s.add_reaction("d2*Skip", {"Skip":-1}, "Skip degr."  )
        s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1}, "Retention" )
        s.add_reaction("d3*ret",  {"ret": -1}, "Ret. degr." )
        
        s.set_cluster("Pr_on", (1,))
        s.set_cluster("Pr_off", (1,))
        
    elif(name == "LotkaVolterra"):
        s = SimParam("Lotka Volterra", 50, 301,
                      {"k1":1, "k2":0.007, "k3":0.6 },
                      {"Prey":50, "Predator":200})
        s.add_reaction("k1*Prey", {"Prey":[-2,3]}, "Reproduction")
        s.add_reaction("k2*Prey*Predator",{"Prey":-1, "Predator":1}, "Hunt")
        s.add_reaction("k3*(Predator)",{"Predator":-1}, "Death")
    elif(name == "CoTrSplicing"):
        
        # https://www.ncbi.nlm.nih.gov/pubmed/15217358
        gene_len = 3000
        u1_1_bs_pos = 150
        u2_1_bs_pos = 1500
        u1_2_bs_pos = 1700
        u2_2_bs_pos = 2800
        
        v0 = 60
        spl_r = 0.8
        
        v1_1 = 0.234
        v2_1 = 0.024
        v1_2 = 0.012
        v2_2 = 1.25
        s2 = 1/(3/(v1_1+v2_2)  + 1/(spl_r))
        s1 = 1/(3/(v1_1+v2_1)  + 3/(v1_2+v2_2) + 2/(spl_r))
        
        # consider https://science.sciencemag.org/content/sci/331/6022/1289/F5.large.jpg?width=800&height=600&carousel=1
        #for Ux binding rates
        params = {"pr_on": 2, "pr_off" : 0.1,
                "elong_v": v0, # 20-80 nt per second . http://book.bionumbers.org/what-is-faster-transcription-or-translation/
                "gene_len": gene_len,
                "spl_rate": spl_r,#0.002, # 1/k3 = 1/k2 + 1/k1
                "u1_1_bs_pos": u1_1_bs_pos , # U1 binding site position
                "u1_2_bs_pos": u1_2_bs_pos ,
                "u2_1_bs_pos": u2_1_bs_pos, # U2 bind. site pos 1
                "u2_2_bs_pos": u2_2_bs_pos,
                "u1_1_br": v1_1,  # binding rate of U1 
                "u2_1_br": v2_1,
                "u1_2_br": v1_2,  # binding rate of U1
                "u2_2_br": v2_2,
                "u1ur": 0.001,  # unbinding rate of U1 
                "u2ur": 0.001, # unbinding rate of U1
#                "tr_term_rate": 100,
#                "s1":s1, "s2":s2, "s3": 1e-4,
                # http://book.bionumbers.org/how-fast-do-rnas-and-proteins-degrade/
                "d1": 2e-4, "d2":2e-4, "d3":1e-3 # mRNA half life: 10-20 h -> lambda: math.log(2)/hl
                }
        
        
        s = SimParam("Cotranscriptional splicing", 10000, 10001, params = params,
                        init_state = {"Pol_on":0, "Pol_off": 1,
                                      "nascRNA_bc": 0,
                                      "Pol_pos": 0,
                                      "Skip":0, "Incl": 0, "ret": 0,
                                      "U1_1":0, "U1_2":0,   #Splicosome units U1 binded
                                      "U2_1":0, "U2_2":0,
                                      "Intr1":0, "Intr2":0, "Exon1":0,
                                      "U11p":0, "U21p":0, "U12p":0, "U22p":0})
        # for drawing PN with SNAKES
        [s.set_cluster(sp,(0,)) for sp in ["Incl", "Skip"]]
        [s.set_cluster(sp,(1,)) for sp in ["ret", "ret_i1", "ret_i2"]]
        [s.set_cluster(sp,(2,)) for sp in ["Intr1", "Intr2", "Exon1"]]
        [s.set_cluster(sp,(3,)) for sp in ["Intr1_ex", "Intr2_ex", "Exon1_ex"]]
        [s.set_cluster(sp,(4,)) for sp in ["Pol_on", "Pol_off", "Pol_pos"]]
        [s.set_cluster(sp,(5,)) for sp in ["U1_1", "U1_2", "U2_1", "U2_2"]]
        ###########################
        
        s.add_reaction("pr_on * Pol_off",
                       {"Pol_on":1, "Pol_off": -1,"Exon1":1, "Intr1":1,"Intr2":1,
                        "Tr_dist": "gene_len", "nascRNA_bc": "-nascRNA_bc"},
                       name = "Transc. initiation")
        
        s.add_reaction("elong_v * Pol_on",
                       {"nascRNA_bc": 1, "Pol_pos":1, "Tr_dist":-1},
                       name = "Elongation")
        
        # Ux (un)binding cinetics
        s.add_reaction("u1_1_br * Intr1",
                       {"U1_1":[1,None], "Intr1": [-1,1],
                        "Pol_pos":["-u1_1_bs_pos", "u1_1_bs_pos"]},
                       "U1_1 binding")
        s.add_reaction("u1ur * U1_1", {"U1_1":-1}, "U1_1 diss.")
        
        s.add_reaction("u1_2_br * Intr2",
                       {"U1_2":[1,None], "Intr2": [-1,1],
                        "Pol_pos":["-u1_2_bs_pos", "u1_2_bs_pos"]},
                        "U1_2 binding")
        s.add_reaction("u1ur * U1_2", {"U1_2":-1}, "U1_2 diss.")
        
        s.add_reaction("u2_1_br * Intr1",
                      {"U2_1":[1,None], "Intr1": [-1,1],
                        "Pol_pos":["-u2_1_bs_pos", "u2_1_bs_pos"]},
                       "U2_1 binding")
        s.add_reaction("u2ur * U2_1", {"U2_1":-1}, "U2_1 diss.")
        
        s.add_reaction("u2_2_br * Intr2",
                       {"U2_2":[1,None], "Intr2": [-1,1],
                        "Pol_pos":["-u2_2_bs_pos", "u2_2_bs_pos"]},
                        "U2_2 binding")
        s.add_reaction("u2ur * U2_2", {"U2_2":-1}, "U2_2 diss.")
        
        #Splicing
        s.add_reaction("U1_1 * U2_1 * Intr1 * spl_rate",
                       {"Intr1":-1, "U1_1":-1, "U2_1":-1,
                        "nascRNA_bc": "-(u2_1_bs_pos - u1_1_bs_pos)"},
                       name="Intron 1 excision")
        s.add_reaction("U1_2 * U2_2 * Intr2 * spl_rate",
                       {"Intr2":-1, "U1_2":-1, "U2_2":-1,
                        "nascRNA_bc": "-(u2_2_bs_pos - u1_2_bs_pos)"},
                       name="Intron 2 excision")
        s.add_reaction("U1_1 * U2_2 * spl_rate",
                       {"Intr1":-1, "Intr2":-1, "Exon1":-1,
                        "nascRNA_bc": "-(u2_2_bs_pos - u1_1_bs_pos)",
                        "U1_1":-1, "U2_2":-1, "U1_2":"-U1_2", "U2_1":"-U2_1"},
                       name="Exon 1 excision (inclusion)")
        
        #Transcription termination
        s.add_reaction("elong_v",
                       {"Intr1":None, "Intr2":None, "Exon1":None,
                        "Skip":1, "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1},
                       name = "Termination: skipping")
        s.add_reaction("elong_v",
                       {"Intr1":None, "Intr2":None, "Exon1":-1, "Incl":1,
                        "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1},
                       name = "Termination: inclusion")
        s.add_reaction("elong_v",
                       {"Exon1":-1, "Intr1":-1, "Intr2":None, "ret_i1":1,
                        "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1,
                        "U1_1":"-U1_1", "U2_1": "-U2_1",
                        "U11p":"U1_1", "U21p":"U2_1"},
                       name = "Termination: ret i1")
        s.add_reaction("elong_v",
                       {"Exon1":-1, "Intr2":-1, "Intr1":None, "ret_i2":1,
                        "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1,
                        "U1_2":"-U1_2", "U2_2": "-U2_2",
                        "U12p":"U1_2", "U22p":"U2_2"},
                       name = "Termination: ret i2")
        s.add_reaction("elong_v",
                       {"Exon1":-1, "Intr1":-1, "Intr2":-1, "ret":1,
                        "Pol_pos": "-gene_len", "Pol_on":-1, "Pol_off":1,
                        "U1_1":"-U1_1", "U2_1": "-U2_1","U1_2":"-U1_2", "U2_2": "-U2_2",
                        "U11p":"U1_1", "U21p": "U2_1","U12p":"U1_2", "U22p": "U2_2"},
                       name = "Termination: full retention")
         
        #Posttranscriptional reactions        
        s.add_reaction("(ret+ret_i1-U11p)*u1_1_br", {"U11p": 1}, "U11p binding")
        s.add_reaction("U11p*u1ur", {"U11p": -1}, "U11p unbinding")
        s.add_reaction("(ret+ret_i1-U21p)*u2_1_br", {"U21p": 1}, "U21p binding")
        s.add_reaction("U21p*u2ur", {"U21p": -1}, "U21p unbinding")
        s.add_reaction("(ret+ret_i2-U12p)*u1_2_br", {"U12p": 1}, "U12p binding")
        s.add_reaction("U12p*u1ur", {"U12p": -1}, "U12p unbinding")
        s.add_reaction("(ret+ret_i2-U22p)*u2_2_br", {"U22p": 1}, "U22p binding")
        s.add_reaction("U22p*u2ur", {"U22p": -1}, "U22p unbinding")
        
        s.add_reaction("spl_rate*U11p * U21p*ret/((ret+ret_i1)**2)",
                       {"ret": -1, "ret_i2": 1, "U11p":-1, "U21p":-1}, "PostTr. ret -> ret_i2")
        s.add_reaction("spl_rate*U11p * U21p*ret_i1/((ret+ret_i1)**2)",
                       {"ret_i1": -1, "Incl": 1, "U11p":-1, "U21p":-1}, "PostTr. ret_i1 -> Incl")
        s.add_reaction("spl_rate*U12p * U22p*ret/((ret+ret_i2)**2)",
                       {"ret": -1, "ret_i1": 1, "U12p":-1, "U22p":-1}, "PostTr. ret -> ret_i1")
        s.add_reaction("spl_rate*U12p * U22p*ret_i2/((ret+ret_i2)**2)",
                       {"ret_i2": -1, "Incl": 1, "U12p":-1, "U22p":-1}, "PostTr. ret_i2 -> Incl")
        
        s.add_reaction("spl_rate * U11p*ret/(ret+ret_i1) * U22p/(ret+ret_i2)",
                       {"ret": -1, "Skip": 1, "U11p":-1, "U22p":-1,
                        "U21p":"-round(U21p/(ret+ret_i1) if U21p > 0 else 0)",
                        "U12p":"-round(U12p/(ret+ret_i2) if U12p > 0 else 0)"},
                       "PostTr. ret -> Skip")
#        s.add_reaction("((u1_1_br+u2_2_br)/2  + 1/(spl_rate)) * ret", {"ret": -1, "Skip": 1}, "PostTr. ret -> Skip")
#        s.add_reaction("((u1_1_br+u2_1_br)/2  + 1/(spl_rate)) * ret", {"ret": -1, "ret_i2": 1}, "PostTr. ret -> ret_i2")
#        s.add_reaction("((u1_2_br+u2_2_br)/2  + 1/(spl_rate)) * ret", {"ret": -1, "ret_i1": 1}, "PostTr. ret -> ret_i1")
#        s.add_reaction("((u1_2_br+u2_2_br)/2  + 1/(spl_rate)) * ret_i2", {"ret_i2": -1, "Incl": 1}, "PostTr. ret_i2 -> Incl")
#        s.add_reaction("((u1_1_br+u2_1_br)/2  + 1/(spl_rate)) * ret_i1", {"ret_i1": -1, "Incl": 1}, "PostTr. ret_i1 -> Incl")
        #Degradation
        s.add_reaction("d1 * Incl", {"Incl": -1}, "Incl degr.")
        s.add_reaction("d2 * Skip", {"Skip": -1}, "Skip degr.")
#        s.add_reaction("s3 * mRNA", {"mRNA": -1, "ret": 1})
#        s.add_reaction("d3 * ret", {"ret": -1}, "ret degr.")
#        s.add_reaction("d3 * ret_i1", {"ret_i1": -1}, "ret_i1 degr.")
#        s.add_reaction("d3 * ret_i2", {"ret_i2": -1}, "ret_i2 degr")
#        
        
    elif(name == "CoTrSplicing_2"):
        s = get_exmpl_sim("CoTrSplicing")
        s.set_param("u2_pol_br", 1) #binding rate of U2 + Pol
        s.set_param("u2_pol_ur", 0.01)
        s.set_param("u2pol_br", 1) #max binding rate of U2Pol + mRNA
        s.set_param("u2_pol_opt_d", 20) # optimal distance from Pol2
        s.set_param("u2_pol_opt_d_r", 10)
        s.add_reaction("Pol_on * u2_pol_br", {"U2_Pol":[1, None]}, "Pol + U2")
        s.add_reaction("U2_Pol * u2_pol_ur", {"U2_Pol":-1}, "Pol/U2 diss.")
        s.add_reaction("(U2_Pol * Intr1 * u2pol_br * (1-1/(1 + u2_pol_opt_d_r /abs(Pol_pos - u2_1_bs_pos+0.01))**4))",
                        {"U2_1":[1,None], "U2_Pol":-1, "Pol_pos":["u2_1_bs_pos", "-u2_1_bs_pos"]},
                        "U2onPol to U2_1")
        s.add_reaction("U2_Pol * Intr2 * u2pol_br * (1-1/(1 + u2_pol_opt_d_r /abs(Pol_pos - u2_2_bs_pos+0.01))**4) ",
                        {"U2_2":[1,None], "U2_Pol":-1, "Pol_pos":["u2_2_bs_pos", "-u2_2_bs_pos"]},
                        "U2onPol to U2_2")
        
#        s.add_reaction()
        
    return s

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

            