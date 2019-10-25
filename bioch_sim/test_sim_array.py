#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:25:55 2019

@author: timur
"""

import bioch_sim as bs
import numpy as np
import numba as nb
import matplotlib.pylab as plt

import time

s = bs.get_exmpl_sim()
s = bs.get_exmpl_sim("CoTrSplicing_2")
s.simulate(200)
#s.simulate()
#s.plot_course(clear=True)
lr = s._last_results

ax = s.plot_series(products = ["Incl", "Skip", "U2_1"])
ax.legend()