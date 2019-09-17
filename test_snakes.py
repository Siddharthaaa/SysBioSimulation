# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:28:59 2019

@author: Timur
"""

import snakes
import snakes.plugins as snpi
snpi.load("gv","snakes.nets","nets")
import nets as ns

import bioch_sim as bs

#sys.path.append("C:\Program Files (x86)\Graphviz2.38\\bin")
pn = ns.PetriNet("mynet")
pn.add_place(ns.Place("p1",[ns.dot]))
pn.add_place(ns.Place("p2", [ns.dot]))
pn.add_transition(ns.Transition("t"))
pn.add_input("p1", "t", ns.Value(ns.dot))
pn.add_input("p2", "t", ns.Value(ns.dot))
pn.add_output("p2", "t", ns.Value(ns.dot))

pn.draw("SNAKES_Test.png")

s = bs.get_exmpl_sim("CoTrSplicing")
s.compile_system()
s.draw_pn(engine="dot", debug=True)

for engine in ('neato', 'dot', 'circo', 'twopi', 'fdp', "sfdp" ) :
    s.draw_pn('test-gv-%s.png' % engine, engine=engine)
#    pn.draw('test-gv-%s.png' % engine, engine=engine)
    
s = bs.get_exmpl_sim("hill_fb")
s = bs.get_exmpl_sim("LotkaVolterra")
s = bs.get_exmpl_sim("CoTrSplicing")
for sp in ["Incl", "Skip", "ret" ]:
    s._clusters[sp] = (1,)
for sp in ["U1_1", "U1_2", "U2_1", "U2_2" ]:
    s._clusters[sp] = (0,)

s = bs.get_exmpl_sim()
s.draw_pn(rates=False, engine="dot")
