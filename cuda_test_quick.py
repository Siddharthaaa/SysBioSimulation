# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:35:42 2019

@author: Timur
"""

import bioch_sim as bs
import numba as nb
import numpy as np 
import time

n = 20


sim = bs.get_exmpl_sim()
sim.set_runtime(1e3)
sim.set_raster(101)
sim.simulate_ODE = False
start = time.time()
for i in range(n**2):
#    sim.simulate()
    pass
#sim = bs.get_exmpl_sim("CoTrSplicing")
end = time.time() -start
print("RUNTIME: ", end)

elong_v = np.ones(n) * 60
spl_rate = np.ones(n) * 60

v_syn = np.ones(n)* 100
s1 = np.ones(n) * 1

sim.params
sim.simulate()
start = time.time()
res = sim.simulate_cuda({"v_syn": v_syn, "s1": s1})
end = time.time() -start
print("RUNTIME: ", end)
#sim.simulate_cuda({"elong_v": elong_v, "spl_rate": spl_rate})


#sim.plot_cuda()
