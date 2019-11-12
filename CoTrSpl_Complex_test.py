#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:00:31 2019

@author: timur
"""

import numpy as np
import bioch_sim as bs
import pylab as plt


s = bs.get_exmpl_sim("CoTrSplicing_2")
s.set_runtime(1e5)
s.simulate(tr_count=30)
#s.show_interface()
s.plot_series(products=["Incl", "Skip"])
