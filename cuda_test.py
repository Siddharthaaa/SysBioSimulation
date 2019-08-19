# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:41:21 2019

@author: Timur
.."""

from pylab import imshow, show

from numba import cuda
import bioch_sim as bs
from timeit import default_timer as timer
import numpy as np
import math


#s1= s2 = s3 = d1 = d2 = d3 = d0 = 1
#s1 = .1
#d1 = 0.2
#d2 = 0.3
#d3 = 1.5
#k_syn = 100
#
#name="Test"
#s = bs.SimParam(name,100, 1001,
#             params = {"v_syn": k_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
#             init_state = {"pre_RNA": 10, "Incl": 3, "Skip":2, "ret": 1})
#
#s.simulate_ODE = False
#
#s.add_reaction("v_syn", {"pre_RNA":1} )
#s.add_reaction("d0*pre_RNA", {"pre_RNA":-1} )
#s.add_reaction("s1*pre_RNA", {"pre_RNA":-1, "Incl":1})
#s.add_reaction("d1*Incl", {"Incl": -1}  )
#s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1})
#s.add_reaction("d2*Skip", {"Skip":-1}  )
#s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1} )
#s.add_reaction("d3*ret",  {"ret": -1} )
#
#
#s.add_reaction("d3*ret",  {"ret": -1} )
#s.add_reaction("d3*ret",  {"ret": -1} )

k_on=0
k_off=0.00
v_syn=10.000
s1=0.500
Ka1=60
n1=6.000
s2=1.000
s3=0.100
d0=0.100
d1=0.100
d2=0.100
d3=0.500
s1_t=1

name="Ka1=%2.2f,s1_t =%2.2f, n1=%2.2f" % (Ka1, s1_t, n1)
s = bs.SimParam(name, 2000, 10001,
             params = {"k_on": k_on, "k_off": k_off, "v_syn": v_syn, "s1": s1, "Ka1":Ka1, "n1":n1,
                       "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3, "s1_t":s1_t},
             init_state = {"Pr_on": 1, "Pr_off": 0, "pre_RNA": 0,
                           "Incl": 10, "Skip": 0, "ret": 0})

s.simulate_ODE = True

s.add_reaction("k_on*Pr_off", {"Pr_on":1, "Pr_off":-1})
s.add_reaction("k_off*Pr_on", {"Pr_off":1, "Pr_on":-1})
s.add_reaction("v_syn*Pr_on", {"pre_RNA":1} )
s.add_reaction("d0*pre_RNA", {"pre_RNA":-1} )
s.add_reaction("s1*pre_RNA + pre_RNA* s1_t * (1/(1 + (Ka1/Incl)**n1) if Incl > 0 else 0)", {"pre_RNA":-1, "Incl":1})
s.add_reaction("d1*Incl", {"Incl": -1}  )
s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1})
#            s.add_reaction("s2*pre_RNA + 5* (Skip > 0)* 1/(1+(Ka2/Skip)**n2)" ,  {"pre_RNA":-1, "Skip":1})
s.add_reaction("d2*Skip", {"Skip":-1}  )
s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1} )
s.add_reaction("d3*ret",  {"ret": -1} )

for i in range(100):
    s.add_reaction("s1*pre_RNA + pre_RNA* s1_t * (1/(1 + (Ka1/Incl)**n1) if Incl > 0 else 0)", {"pre_RNA":-1, "Incl":1})


s.simulate()
print(s.results)
start = timer()
out = s.simulate_cuda(params = {"Ka1": np.linspace(50,70,16),
                                "s1_t": np.linspace(1, 5,16) },  max_steps=100)
dt = timer() -start
print(dt)
cuda_log = s.cuda_log

s.plot_cuda()

s.results["stoch_rastr"] = s.cuda_out[0,1]

s.plot_course()

#print("TESTETST")
s.simulate_cuda()
#out = s._create_out_array()
#print("TESTETST")
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
#res = s.simulate_cuda()
res1 = res[0,0]
res2 = res[1,1]
#print(rates)
s.plot()
r = s.results["stoch_rastr"]
s.results["stoch_rastr"] = res1
s.plot()

sims_n = 32

start = timer()
for i in range(sims_n**2):
    s.simulate()
dt = timer() - start
print(dt, " ss")

start = timer()
res10 = s.simulate_cuda(params = {"s1": np.ones(sims_n, np.float32),
                                  "s2": np.random.uniform(size=sims_n)+1}, max_steps=1000)
last_params = s.cuda_last_params
dt = timer() - start
print(dt, " ssss")
