# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:41:21 2019

@author: Timur
"""

from pylab import imshow, show
import pycuda as pc
from numba import cuda
import bioch_sim as bs
from timeit import default_timer as timer
import numpy as np
import math



s1= s2 = s3 = d1 = d2 = d3 = d0 = 1
s1 = .1
d1 = 0.2
d2 = 0.3
d3 = 1.5
k_syn = 100

name="Test"
s = bs.SimParam(name,200, 1001,
             params = {"v_syn": k_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
             init_state = {"pre_RNA": 0, "Incl": 0, "Skip": 0, "ret": 0})

s.simulate_ODE = False

s.add_reaction("v_syn", {"pre_RNA":1} )
s.add_reaction("d0*pre_RNA", {"pre_RNA":-1} )
s.add_reaction("s1*pre_RNA", {"pre_RNA":-1, "Incl":1})
s.add_reaction("d1*Incl", {"Incl": -1}  )
s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1})
s.add_reaction("d2*Skip", {"Skip":-1}  )
s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1} )
s.add_reaction("d3*ret",  {"ret": -1} )

s.simulate()
#sims_n = 4
#res10 = s.simulate_cuda(params = {"s1": np.ones(sims_n, np.float32),
#                                  "s2": np.random.uniform(size=sims_n)+1})

#s.plot()

def test_func():
    return 1

gpu_func = cuda.jit(device=True)(s._rates_function)

cuda.jit(bs.compute_stochastic_evolution, device=True)

@cuda.jit
def test_func_ref(a):
    return np.log(a)

@cuda.jit
def get_cuda_infos(out, st, par, r_rates ):
#    stat =  cuda.local.array(len(st), np.int32)
#    stat = st
    X, Y = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    x = cuda.blockIdx.x 
    x1 = cuda.blockIdx.y
    y = cuda.threadIdx.x
    y1 = cuda.threadIdx.y
#    stat[1] = 200
    xx = gpu_func(st, par[X,Y], r_rates[X,Y])
    
#    out[startX, startY] = gridY
#    out[X, Y] = math.log(r_rates[X,Y][3] + r_rates[X,Y][0])
    for i in range(len(xx)):
        out[X,Y,i] = xx[i]
#    print(startX, startY)

out = np.zeros((100, 200,8), dtype = np.float64)
#blockdim = (32, 8)
blockdim = (10,20)
#griddim = (32,16)
griddim = (10, 10)

start = timer()
d_image = cuda.to_device(out)
st = np.array([0,10,10,9,5,7])
par =  np.random.rand(blockdim[0]*griddim[0], blockdim[1]*griddim[1], 8) * 5
react_rates= np.zeros((blockdim[0]*griddim[0], blockdim[1]*griddim[1],len(s._reactions)), np.float64)
d_st = cuda.to_device(st)
d_par = cuda.to_device(par)
d_react_rates = cuda.to_device(react_rates)
get_cuda_infos[griddim, blockdim](d_image, d_st, d_par, d_react_rates) 
d_image.to_host()
d_react_rates.to_host()
dt = timer() - start

imshow(out[:,:,1])

s._rates_function(st, par[0,7], react_rates[0,0])
print(s.compile_system(dynamic=True))

#res = s.simulate_cuda({"s1": [1,2,3,4]}, max_steps = 100000)
res = s.simulate_cuda( max_steps = 100000)
res1 = res[0,0]
res2 = res[1,1]
#print(rates)
s.plot()
r = s.results["stoch_rastr"]
s.results["stoch_rastr"] = res1
s.plot()

sims_n = 20

start = timer()
for i in range(sims_n):
    s.simulate()
dt = timer() - start
print(dt, " s")

start = timer()
res10 = s.simulate_cuda(params = {"s1": np.ones(sims_n, np.float32),
                                  "s2": np.random.uniform(size=sims_n)+1})
dt = timer() - start
print(dt, " s")
