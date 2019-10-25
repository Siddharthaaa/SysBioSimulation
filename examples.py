# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:19:49 2019

@author: Timur
"""


import bioch_sim as bs
import matplotlib.pyplot as plt
import support_th as sth

import numpy as np

s = bs.get_exmpl_sim()
s.simulate()

s = bs.SimParam("2A + B <=> X", 1.5,10001,
                dict(k1=0.01, k2=0.05),
                dict(A=200, B = 450, X = 0))
s.add_reaction("k1*A*A*B", {"A":-2, "B":-1, "X":1}, "forward")
s.add_reaction("k2*X", {"A":2, "B":1, "X":-1}, "back")
s.simulate()
s.plot_course()
s.set_cluster("A", (1,))
s.set_cluster("B", (1,))
s._set_color("A","orange")
s._set_color("B","blue")
s._set_color("X","green")
s.draw_pn(filename="2ABX.png"  , engine="dot", rates = True)



s = bs.SimParam("2A + B <=> X", 1.5,10001,
                dict(k1=0.01, k2=0.05),
                dict(A=200, B = 450, X = 0))
s.add_reaction("k1*A*A*B", {"A":-2, "B":-1, "X":1, "Enz":[-1,1]}, "forward")
s.add_reaction("k2*X", {"A":2, "B":1, "X":-1}, "back")
s.simulate()
s.plot_course()
s.set_cluster("A", (1,))
s.set_cluster("B", (1,))
s._set_color("A","orange")
s._set_color("B","blue")
s._set_color("X","green")
s.draw_pn(filename="2ABX_Enz.png"  , engine="dot", rates = True)

s = bs.get_exmpl_sim("hill_fb")
s.draw_pn(rates = False, engine = "dot")


s = bs.SimParam("2A + B <=> X", 1.5,10001,
                dict(k1=0.01, k2=0.05),
                dict(A=200, B = 450, X = 0, Light=0))
s.add_reaction("k1*A*A*B", {"A":-2, "B":-1, "X":1, "Enzyme":[-1,1]}, "forward")
s.add_reaction("k2*X", {"A":2, "B":1, "X":-1, "Light":None}, "back")
s.simulate()
s.plot_course()
s._set_color("A","orange")
s._set_color("B","blue")
s._set_color("X","green")

#s.show_interface()
#s.simulate_ODE = True
s.draw_pn(filename="2ABX_Enz_Light.png"  , engine="dot", rates = True)


s = bs.SimParam("Transcription_low", 100,10001,
                dict(v_init=1, v_elong=50, v_term=1e-4),
                dict(Pol=0, nascRNA = 0))
s.add_reaction("v_init", {"Pol":[1,None] }, "Initiation")
s.add_reaction("v_elong", {"nascRNA":1, "Pol":[1,-1]}, "Elongation")
s.add_reaction("v_term*nascRNA", {"nascRNA":0, "Pol":-1}, "Termination")
s.simulate()
s.plot_course(products="Pol", products2="nascRNA")
s._set_color("Pol","orange")
s._set_color("nascRNA","blue")
#s._set_color("X","green")

s.show_interface()
#s.simulate_ODE = True
s.draw_pn(engine="dot", rates = True)