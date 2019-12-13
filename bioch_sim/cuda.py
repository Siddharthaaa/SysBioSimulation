#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:24:46 2019

@author: timur
"""

import math
import numba as nb
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64

class SimParam(SimParam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
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