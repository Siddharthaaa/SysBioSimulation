# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:45:28 2019

@author: imb30
"""

from tkinter import Tk # copy to clipboard function
from tkinter import filedialog
import tkinter as tk
from tkinter.colorchooser import askcolor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
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
import pickle
import re
import numba as nb
from sklearn.cluster import KMeans, MeanShift, k_means
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64

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

class SimParam(object):
    def __init__(self, name, t=200, discr_points= 1001, params={}, init_state={}):
        self.name = str(name)
        self.runtime = t
        self.set_raster_count(discr_points)
        self.params=params
        
        self.init_state=init_state
        self.simulate_ODE = False
        self._rate_funcs =[]
        self._reactions =[]
        self._transitions = []
        self.id=id(self)
        self._is_compiled = False
        self._dynamic_compile = False
        self._clusters ={}
        self._reset_results()
    def set_param(self, name, value):
        self.params[name] = value
        self._is_compiled = self._dynamic_compile
        self._reset_results()
    def _reset_results(self):
        self.bimodality = {}    
        self.results={}
    def set_raster_count(self, discr_points):
        self.raster_len = discr_points
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
        self.set_raster_count(self.raster_len)
        
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
        
        self._transitions.append(dict(name=name, rate=rate_func,
                                      actors = reaction))
        
        for name in reaction.keys():
            if not name in self.init_state.keys():
                self.init_state[name] = 0
        self._is_compiled = False
#    @nb.jit  # causes performance drops
    def get_rates(self, state=None):
        """ state must contain time as first element
        """
        if state is None:
            state=self._state
        return self._rates_function(state, self._constants, np.zeros(len(self._reactions)))
    def get_reacts(self, state = None):
        #returns reaction matrix
        if state is None:
            state = self._state
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
                engine=('neato', 'dot', 'circo', 'twopi', 'fdp'), **kwargs):
        if type(engine) is str:
            engine = (engine,)
        self.compile_system()
        #https://www.ibisc.univ-evry.fr/~fpommereau/SNAKES/API/plugins/gv.html
        if filename is None:
            filename = self.name + ".png"
        
        pn = pns.PetriNet(self.name)
        for p, v in self.init_state.items():
            cluster = self._clusters[p] if p in self._clusters else ()
            pn.add_place(pns.Place(p,v), cluster=cluster)
        
        for tr in self._transitions:
            name = tr["name"]
            cluster = self._clusters[name] if name in self._clusters else ()
            if(rates):
                pn.add_transition(pns.Transition(name, pns.Expression(tr["rate"])),
                                                 cluster = cluster)
            else:
                pn.add_transition(pns.Transition(name), cluster=cluster)
            for p, vs in tr["actors"].items():
                for v in vs:
                    if(v > 0):
                        pn.add_output(p, name, pns.Value(v))
                    else:
                        pn.add_input(p, name, pns.Value(-v))
        
        
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
        if(rotation):
            pn.transpose()
        for e in engine:
            f_name = re.sub("(\.\w+)$", "_"+ e + "\\1", filename)
            f_name = os.path.join("PN_images", f_name)
            pn.draw(f_name, engine = e, place_attr=draw_place,
                trans_attr=draw_transition , **kwargs)
        return pn
    
    def compile_system(self, dynamic = True):
        #create reaction matrix
        self._reacts = np.zeros(( len(self._reactions), len(self.init_state)), dtype=int )
        self._pre = self._reacts.copy()
        self._post = self._reacts.copy()
        self._state = list(self.init_state.values())
        self._state.insert(0,0)
        self._state = np.array(self._state, dtype=np.float64)
        names = list(self.init_state.keys())
        i=0
        for react in self._reactions:
            for substance in react.keys():
                for v in react[substance]:
                    if v < 0:
                        self._pre[i, names.index(substance)] += -v
                    else:
                        self._post[i, names.index(substance)] += v
#                self._reacts[i, names.index(substance)] += sum(react[substance])
            i+=1
        self._reacts = self._post - self._pre
        
        #create rates function
        self._r_s = np.zeros((len(self._rate_funcs),))
        func_str= "@nb.njit\n"
#        func_str= ""
        
        func_str+= "def _r_f_(st, constants, _r_s):\n"
#        func_str+= "\t_r_s=np.zeros(%d)\n" % len(self._r_s)
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
    def simulate(self, ODE = False, ret_raw=False, max_steps = 1e10):
        if not self._is_compiled:
            self.compile_system(dynamic=True)
        self._state = list(self.init_state.values())
        self._state.insert(0,0)
        self._state = np.array(self._state, dtype=np.float64)
        print("simulate " + self.param_str())
        results={}
        params = self.get_all_params()
        initial_state = params["init_state"]
        tt = params["raster"]
#        print("raster:" ,len(tt))
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
            @cuda.jit
            def compute_stochastic_evolution_cuda(STATES, reacs, rates_b, constants_all,
                                                  time_steps, max_steps, rng_st, progr_i, log):

                th_nr = cuda.grid(2)
                #TODO 
                x, y = th_nr[0], th_nr[1]
                thid = cuda.blockDim.x * y + x
                
                STATES = STATES[x,y]
                constants_all = constants_all[x,y]
                rates_b = rates_b[x,y]
               
                steps = nb.int32(0)
                t_ind = progr_i[x,y]
#                t_log[x,y] = t_ind
                length = len(time_steps)
                tt=time_steps[t_ind]
                
                
                
                while steps < max_steps and t_ind < length:
#                    tmp = 0
#                    for ST in STATES[t_ind]:
#                        if(ST<0 and log[x,y] == 0):
#                            log[x,y] =  1
#                    tmp +=1
#                    tmp = 0
#                    for const in constants_all:
#                        if(const<0 and log[x,y] == 0):
#                            log[x,y] =  2
#                    tmp +=1
                        
                    r1 = xoroshiro128p_uniform_float32(rng_st, thid*2)
                    r2 = xoroshiro128p_uniform_float32(rng_st, thid*2+1)
                    gpu_rates_func(STATES[t_ind], constants_all, rates_b)
#                    tmp = 0
#                    for r in rates_b:
#                        if(r<0 and log[x,y] == 0):
#                            log[x,y] =  3
#                        tmp +=1

                    a_0 =0
                    for i in range(len(rates_b)):
                        a_0 += rates_b[i]
#                        if(rates_b[i]<0 and log[x,y] == 0):
#                            log[x,y] = i
#                    if a_0 <0:
#                        None
#                        log[x,y] = 3
                    
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
                    
#                    if ((ind == 1 or ind==0) and log[x,y] == 0):
#                        log[x,y] = t_ind
                        
                    # update the systems state
                    for j, r in enumerate(reacs[ind]):
                        STATES[t_ind][j+1] += r
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
                                             d_all_params, raster, max_steps,
                                             rng_states, d_progr_indx, d_log_arr )
#            print(test)
#            print("end loop ", i)
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
   
    def get_res_from_expr(self, expr):
#        for k,v  in self.init_state.items():
#            expr = re.sub("\\b" + k + "\\b", "self.get_res_col(\"%s\")" % k, expr)
#        
#        for k,v  in self.params.items():
#            expr = re.sub("\\b" + k + "\\b", "%e" % v, expr)
#        print("try to eval: " + expr)
#        res = eval(expr)
#        
        args = []
        arg_vals = []
        for k,v  in self.init_state.items():
            if(re.search("\\b" + k + "\\b", expr) is not None):
                args.append(k)
                arg_vals.append(self.get_res_col(k))
#        print(args)
        for k,v  in self.params.items():
            expr = re.sub("\\b" + k + "\\b", "%e" % v, expr)                
        expr = "lambda " + ", ".join(args) + ": ("   + expr +")"
        f = eval(expr)
        f = np.vectorize(f, otypes=[np.float] )
        res = f(*arg_vals)
        
        return res
        
    def get_psi_cv(self, **kwargs):
        psi = self.compute_psi(**kwargs)[1]
        sd, mean = np.std(psi), np.mean(psi)
        return sd/mean
    def get_psi_mean(self, **kwargs):
        self.compute_psi(**kwargs)
        return np.mean(self.results["PSI"][1])
    def get_res_index(self, name):
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
    
    def plot_course(self, ax = None, res=["ODE","stoch"], products=[], rng = None,
                    line_width=2, scale = 1, plot_mean=False, plot_psi=False):
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
        if(plot_psi):
            ax_psi = ax.twinx()
            (indx, psis) = self.compute_psi()
            ax_psi.plot(self.raster[indx], psis, ".", markersize=line_width*5, label = "PSI")
            ax_psi.set_ylabel("PSI")
        ax.set_ylabel(self.param_str("\n"), rotation=0, fontsize="large" )
#        ax.set_ylabel("#")
        ax.set_xlabel("time",fontsize="large" )
        ax.set_title(self.name)
        ax.legend()
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
    
    def plot_par_var_1d(self, par = "s1", vals = [1,2,3,4,5],ax = None, func=None, **func_pars):
        res = []
        for v in vals:
            self.set_param(par, v)
            self.simulate()
            res.append(func(**func_pars))
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(vals, res)
        ax.set_ylabel(func.__name__ +   str(func_pars))
        ax.set_xlabel(par)
        
        return ax
    
    def plot_par_var_2d(self, pars = {"s1":[1,2,3], "s2": [1,2,4]},ax = None, func=None, **func_pars):
        
        names = list(pars.keys())
        
        res = []
        sim_res =[]
        for par1 in pars[names[0]]:
            r = []
            sim_r = []
            self.set_param(names[0], par1)
            for par2 in pars[names[1]]:
                self.set_param(names[1], par2)
                self.simulate()
                sim_r.append(self.results["stoch_rastr"])
                r.append(func(**func_pars))
            res.append(r)
            sim_res.append(sim_r)
            
        if ax is None:
            fig, ax = plt.subplots()
        
        heatmap(np.array(res), pars[names[0]], pars[names[1]], ax, cbarlabel= func.__func__.__name__ +   str(func_pars) )
        ax.set_xlabel(names[1])
        ax.set_ylabel(names[0])
        
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
        
        self.master.title(sim.name)
        self.pack(fill=tk.BOTH, expand=True)
                
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)
        
        f_places = tk.Frame(self)
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
        
        self._show_sp = []
        i = 0
        p_labels = []
        self.p_entries = {}
        for k, v in sim.params.items():
#            row = tk.Frame(f_params)
#            row.pack(side=tk.TOP, fill=tk.X, padx = 1, pady=1)
            label = tk.Label(f_params, text=k)
            label.grid(row=i, column=0)
            entr = tk.Entry(f_params)
            entr.insert(0,v)
            entr.grid(row=i, column=1)
            entr.bind('<Return>', lambda e: self.update(True) ) 
            self.p_entries[k]= entr
            i +=1
        self._spezies_checkb = {}
        self._spezies_col_b = {}
        i = 0
        for k, v in sim.init_state.items():
            c = sim._get_color(k)
            checked = tk.IntVar()
            check_box = tk.Checkbutton(f_places, text=k, variable = checked, #command = self.update)
                                       command= lambda : self.update(False))
#            check_box.select()
            check_box.grid(row=i, column=0)
            self._spezies_checkb[k] = checked
            entr = tk.Entry(f_places, width=4)
            entr.insert(0,v)
            entr.grid(row=i, column=1)
            self.p_entries[k]= entr
            b_col = tk.Button(f_places, height =1, width=1, bg = colors.to_hex(c),
                              command = lambda key=k: self._new_color(key))
            b_col.grid(row=i, column=3)
            self._spezies_col_b[k] = b_col
            
            i +=1
        
        fig = plt.Figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        self._ax = ax
        canvas = FigureCanvasTkAgg(fig, f_plot)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self._canvas = canvas
        self._plot_sim()
        
        f_control = tk.Frame(self)
        f_control.pack(side =tk.BOTTOM, fill =tk.X)
        b_update = tk.Button(f_control, text="Update",
                             command = lambda: self.update(True))
        b_update.pack(side=tk.RIGHT)
    
    def _new_color(self, name):
        b_c = self._spezies_col_b[name]
        col_new = askcolor(b_c.cget("bg"))[1]
        if col_new is not None:
            b_c.configure(bg=col_new)
            self.sim._set_color(name, col_new)
        self.update(False)
    
    def _plot_sim(self):
        ax = self._ax
        ax.clear()
        print(self._show_sp)
        self.sim.plot_course(ax = ax, products = self._show_sp)
        ax.legend([])
        ax.set_ylabel("#", rotation = 0)
        self._canvas.draw()
    
    def update(self, sim=False):
        self.fetch_places()
        if(sim):
            self.fetch_pars()
            self.sim.simulate()
        print("update..")
        self._plot_sim()
        
    def fetch_places(self):
        sim = self.sim
        self._show_sp = []
        for k, checked in self._spezies_checkb.items():
            if checked.get():
                self._show_sp.append(k)
        
    def fetch_pars(self):
        sim = self.sim
        for k, e in self.p_entries.items():
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
def compute_stochastic_evolution(reactions, state, runtime, rate_func, constants, time_steps, max_steps):
   
    STATE = np.zeros((len(time_steps), len(state)),dtype=np.float64)
#    print(state.dtype)
#    state= np.array(state, dtype = np.int64)
    STATE[0,:] = state
    tf = runtime
    tt = 0
    steps = nb.int64(0)
    i = steps+1
    length = len(time_steps)
    rates_array = np.zeros(len(reactions)) 
    while tt <= tf and steps < max_steps and i < length:
        
        # sample two random numbers uniformly between 0 and 1
        rr = sp.random.uniform(0,1,2)
        a_s = rate_func(state, constants, rates_array)
#        print(a_s)
        a_0 = a_s.sum()
        # time step: Meine Version
        #print(a_0)
        if a_0 == 0:
            tt = time_steps[-1]
        else:
            tt = tt - np.log(1. - rr[0])/a_0
        state[0] = tt
        while(tt >= time_steps[i] and i < length):
            STATE[i,:] = state
            STATE[i,0] = time_steps[i]
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
    print("Steps needed: ", steps)
    return STATE
# define parameters  

def sigmoid(x, mu=0, y_bounds=(0,1), range_95=6):
    y=1/(1+np.exp(6/range_95*(-x+mu)))
    y=y*(y_bounds[1]-y_bounds[0])+y_bounds[0]
    return y

def normalized_distance(x, b, p=2):
    return [1/(1+ (1/(abs(xx)/b))**p) if xx != 0 else 0 for xx in x]


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
        v1 = 0.2
        v2 = 0.2
        spl_r = 0.5
        
        v1_1 = 0.1
        v2_1 = 0.2
        v1_2 = 0.2
        v2_2 = 1
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
                "u1_1_br": v1_1,  # binding rate of U1
                "u1_1_ur": 0.001, # unbinding rate of U1
                "u1_2_br": v1_2,  # binding rate of U1
                "u1_2_ur": 0.001,  # unbinding rate of U1
                "u2_1_bs_pos": u2_1_bs_pos, # U2 bind. site pos 1
                "u2_2_bs_pos": u2_2_bs_pos,
                "u2_1_br": v2_1,
                "u2_2_br": v2_2,
                "u2_1_ur": 0.001,
                "u2_2_ur": 0.001,
                "tr_term_rate": 100,
                "Ux_clear_rate": 1e9,
#                "s1":s1, "s2":s2, "s3": 1e-4,
                # http://book.bionumbers.org/how-fast-do-rnas-and-proteins-degrade/
                "d0":2e-4, "d1": 2e-4, "d2":2e-4, "d3":1e-3 # mRNA half life: 10-20 h -> lambda: math.log(2)/hl
                }
        
        
        s = SimParam("Cotranscriptional splicing", 10000, 10001, params = params,
                        init_state = {"Pol_on":0, "Pol_off": 1,
                                      "nascRNA_bc": 0,
                                      "Pol_pos": 0,
                                      "Skip":0, "Incl": 0, "ret": 0,
                                      "U1_1":0, "U1_2":0,   #Splicosome units U1 binded
                                      "U2_1":0, "U2_2":0,
                                      "Intr1":0, "Intr2":0, "Exon1":0,
                                      "Intr1_ex":0, "Intr2_ex":0, "Exon1_ex":0})
        # for drawing PN with SNAKES
        [s.set_cluster(sp,(0,)) for sp in ["Incl", "Skip"]]
        [s.set_cluster(sp,(1,)) for sp in ["ret", "ret_i1", "ret_i2"]]
        [s.set_cluster(sp,(2,)) for sp in ["Intr1", "Intr2", "Exon1"]]
        [s.set_cluster(sp,(3,)) for sp in ["Intr1_ex", "Intr2_ex", "Exon1_ex"]]
        [s.set_cluster(sp,(4,)) for sp in ["Pol_on", "Pol_off", "Pol_pos"]]
        [s.set_cluster(sp,(5,)) for sp in ["U1_1", "U1_2", "U2_1", "U2_2"]]
        ###########################
        
        s.add_reaction("pr_on * Pol_off if U1_1 + U1_2 + U2_1 + U2_2 < 1 else 0", # ugly workaround with Ux
                       {"Pol_on":1, "Pol_off": -1,"Exon1":1, "Intr1":1,"Intr2":1, "Tr_dist": gene_len},
                       name = "Transc. initiation")
        
        s.add_reaction("elong_v * Pol_on if Pol_pos < gene_len else 0",
                       {"nascRNA_bc": 1, "Pol_pos":1, "Tr_dist":-1},
                       name = "Elongation")
        
        # Ux (un)binding cinetics
        s.add_reaction("u1_1_br * Intr1 if Pol_pos > u1_1_bs_pos and U1_1 < 1 else 0",
                       {"U1_1":1}, "U1_1 binding")
        s.add_reaction("u1_1_ur * U1_1", {"U1_1":-1}, "U1_1 diss.")
        
        s.add_reaction("u1_2_br * Intr2 if Pol_pos > u1_2_bs_pos and U1_2 < 1 else 0",
                       {"U1_2":1}, "U1_2 binding")
        s.add_reaction("u1_2_ur * U1_2", {"U1_2":-1}, "U1_2 diss.")
        
        s.add_reaction("u2_1_br * Intr1 if Pol_pos > u2_1_bs_pos and U2_1 < 1 else 0",
                       {"U2_1":1}, "U2_1 binding")
        s.add_reaction("u2_1_ur * U2_1", {"U2_1":-1}, "U2_1 diss.")
        
        s.add_reaction("u2_2_br * Intr2 if Pol_pos > u2_2_bs_pos and U2_2 < 1 else 0",
                       {"U2_2":1}, "U2_2 binding")
        s.add_reaction("u2_2_ur * U2_2", {"U2_2":-1}, "U2_2 diss.")
        
        #Splicing
        s.add_reaction("U1_1 * U2_1 * Intr1 * spl_rate",
                       {"Intr1":-1, "U1_1":-1, "U2_1":-1, "Intr1_ex":1},
                       name="Intron 1 excision")
        s.add_reaction("U1_2 * U2_2 * Intr2 * spl_rate",
                       {"Intr2":-1, "U1_2":-1, "U2_2":-1, "Intr2_ex":1},
                       name="Intron 2 excision")
        s.add_reaction("U1_1 * U2_2 * Intr1 * Intr2 * Exon1 * spl_rate",
                       {"Intr1":-1, "Intr2":-1, "Exon1":-1, "U1_1":-1, "U2_2":-1, "Exon1_ex":1, "Intr1_ex":1, "Intr2_ex":1},
                       name="Exon 1 excision (inclusion)")
        
        #Transcription termination
        s.add_reaction("tr_term_rate * Exon1_ex * Intr1_ex * Intr2_ex if Tr_dist == 0 else 0",
                       {"Exon1_ex":-1, "Intr1_ex":-1, "Intr2_ex":-1, "Skip":1,
                        "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
                       name = "Termination: skipping")
        s.add_reaction("tr_term_rate * Exon1 * Intr1_ex * Intr2_ex if Tr_dist == 0 else 0",
                       {"Exon1":-1, "Intr1_ex":-1, "Intr2_ex":-1, "Incl":1,
                        "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
                       name = "Termination: inclusion")
        s.add_reaction("tr_term_rate * Exon1 * Intr1_ex * Intr2 if Tr_dist == 0 else 0",
                       {"Exon1":-1, "Intr1_ex":-1, "Intr2":-1, "ret_i2":1,
                        "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
                       name = "Termination: Intron 1 retention")
        s.add_reaction("tr_term_rate * Exon1 * Intr1 * Intr2_ex if Tr_dist == 0 else 0",
                       {"Exon1":-1, "Intr1":-1, "Intr2_ex":-1, "ret_i1":1,
                        "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
                       name = "Termination: Intron 2 retention")
        s.add_reaction("tr_term_rate * Exon1 * Intr1 * Intr2 if Tr_dist == 0 else 0",
                       {"Exon1":-1, "Intr1":-1, "Intr2":-1, "ret":1,
                        "Pol_pos": -gene_len, "Pol_on":-1, "Pol_off":1},
                       name = "Termination: full retention")
        # Free Ux sites after/befor transcription. Very ugly workaround
        s.add_reaction("Pol_off * U1_1 * Ux_clear_rate", {"U1_1":-1}, "Clearing...")
        s.add_reaction("Pol_off * U1_2 * Ux_clear_rate", {"U1_2":-1}, "Clearing...")
        s.add_reaction("Pol_off * U2_1 * Ux_clear_rate", {"U2_1":-1}, "Clearing...")
        s.add_reaction("Pol_off * U2_2 * Ux_clear_rate", {"U2_2":-1}, "Clearing...")
         
#        s.add_reaction("d0 * mRNA", {"mRNA": -1})
        s.add_reaction("1/(3/(u1_1_br+u2_1_br)  + 3/(u1_2_br+u2_2_br) + 2/(spl_rate)) * ret", {"ret": -1, "Incl": 1}, "PostTr. Incl")
        s.add_reaction("1/(3/(u1_1_br+u2_2_br)  + 1/(spl_rate)) * ret", {"ret": -1, "Skip": 1}, "PostTr. Skip")
        s.add_reaction("d1 * Incl", {"Incl": -1}, "Incl degr.")
        s.add_reaction("d2 * Skip", {"Skip": -1}, "Skip degr.")
#        s.add_reaction("s3 * mRNA", {"mRNA": -1, "ret": 1})
        s.add_reaction("d3 * ret", {"ret": -1}, "ret degr.")
        s.add_reaction("d3 * ret_i1", {"ret_i1": -1}, "ret_i1 degr.")
        s.add_reaction("d3 * ret_i2", {"ret_i2": -1}, "ret_i2 degr")
        
        
    elif(name == "CoTrSplicing_2"):
        s = get_exmpl_sim("CoTrSplicing")
        s.set_param("u2_pol_br", 1) # binding rate of U2 + Pol
        s.set_param("u2_pol_ur", 0.01)
        s.set_param("u2pol_br", 1) # binding rate of U2Pol + mRNA
        s.set_param("u2_pol_opt_d", 20) # optimal distance from Pol2
        s.set_param("u2_pol_opt_d_r", 5)
        s.add_reaction("Pol_on * u2_pol_br * (1 - U2_Pol)", {"U2_Pol":1}, "Pol + U2")
        s.add_reaction("U2_Pol * u2_pol_ur", {"U2_Pol":-1}, "Pol/U2 diss.")
        s.add_reaction("(U2_Pol * Intr1 * u2pol_br * (1-1/(1 + u2_pol_opt_d_r /abs(Pol_pos - u2_1_bs_pos))**2)) \
                        if Pol_pos > u2_1_bs_pos and U2_1 < 1 else 0", {"U2_1":1, "U2_Pol":-1}, "U2onPol to nascRNA")
        s.add_reaction("U2_Pol * Intr2 * u2pol_br * (1-1/(1 + u2_pol_opt_d_r /abs(Pol_pos - u2_2_bs_pos))**2) \
                        if Pol_pos > u2_2_bs_pos and U2_2 < 1 else 0", {"U2_2":1, "U2_Pol":-1}, "U2onPol to nascRNA")
        s.add_reaction("Pol_off * U2_Pol * Ux_clear_rate", {"U2_Pol":-1}, "Clearing...")
        
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

            