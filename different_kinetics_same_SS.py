# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:53:32 2019

@author: timuhorn
"""

from bioch_sim import *

s = SimParam(name,10000, 10000,
                         params = {"vsyn1": 1, "s1": 10, "vsyn2": 10, "s2": 1, 
                                   "d1": 0.1, "d2": 0.1, "pd1": 0.1, "pd2": 0.1},
                         init_state = {"RNA1": 0, "RNA2": 0,
                                          "Prot1": 0, "Prot2": 0})
                    
s.simulate_ODE = True

s.add_reaction("vsyn1", {"RNA1":1} )
s.add_reaction("vsyn2", {"RNA2":1} )
s.add_reaction("d1*RNA1", {"RNA1":-1} )
s.add_reaction("d2*RNA2", {"RNA2":-1} )
s.add_reaction("s1*RNA1", {"Prot1":1} )
s.add_reaction("s2*RNA2", {"Prot2":1} )
s.add_reaction("pd1*Prot1", {"Prot1":-1} )
s.add_reaction("pd2*Prot2", {"Prot2":-1} )
    
    
s.simulate()
s.plot()