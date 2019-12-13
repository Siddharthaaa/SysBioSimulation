# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:45:28 2019

@author: imb30
"""

from .base import *
from .aux_funcs import *
import os


# import snakes for Petri Net draw
# needs GraphViz
from .draw_pn_ext import SimParam


#import cuda capabilities
#needs cuda
exec(open(os.path.join(os.path.dirname(__file__), "cuda.py")).read())
#from .cuda import SimParam

#extend SimParam by psi functions
exec(open(os.path.join(os.path.dirname(__file__), "psi_ext.py")).read())


from .examples import *