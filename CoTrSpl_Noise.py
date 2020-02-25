# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:54:52 2020

@author: Timur
"""


import bioch_sim as bs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np
import sympy as sy
from sympy.parsing.sympy_parser import parse_expr


s = bs.get_exmpl_CoTrSpl()
s.set_color("Incl", "green")
s.set_color("Skip", "red")

s.set_runtime(40)
s.simulate(200)

s.plot_series(products=("Incl", "Skip",  "P000", "P001" ))
#s.plot_course(products=["Incl", "Skip",  "P000", "P001" ])

res = s.get_res_by_expr_2("Incl/(Incl+Skip)", series=True )
res = np.array(res)
fig, ax = plt.subplots()
ax.plot(res.T, c="black", lw=0.2)
