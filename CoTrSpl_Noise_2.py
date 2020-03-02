#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:20:11 2020

@author: timur
"""

import scipy as sp
import numpy as np
import pylab as plt

import bioch_sim as bs

nums = bs.get_logNorm_rn(mu, cv,200)

s = bs.get_exmpl_CoTrSpl()
s.plot_par_var_1d("vpol", nums)
prod_to_show = ("Incl", "Skip",  "P000", "P100", "P000_inh", "P100_inh", "ret" )

ax = s.plot_series(products=prod_to_show, scale=0.7)
