# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:26:06 2019

@author: Timur
"""

import bioch_sim as bs
import numpy as np
import numba as nb

s = bs.get_exmpl_sim("CoTrSplicing")
s = bs.get_exmpl_sim("CoTrSplicing")
#s = bs.get_exmpl_sim("test")
#s = bs.get_exmpl_sim("LotkaVolterra")
s.set_runtime(1000)
print(s.compile_system())
s.get_reacts()
s.simulate()

s.plot_course(products=["Incl", "Skip"])
s.show_interface()
