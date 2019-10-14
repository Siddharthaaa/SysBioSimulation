# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:35:42 2019

@author: Timur
"""

import bioch_sim as bs
import numba as nb
import numpy as np 

sim = bs.get_exmpl_sim()
sim = bs.get_exmpl_sim("CoTrSplicing")
sim.params
sim.simulate()
sim.simulate_cuda({"v_syn": [20, 30, 40, 50], "s1": [0.5, 1, 2, 4]})

sim.plot_cuda()