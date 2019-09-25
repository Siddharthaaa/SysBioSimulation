#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:49:58 2019

@author: timur
"""

import bioch_sim as bs
import numpy as np
import numba as nb

k1 = 1
s = bs.SimParam("TestChainReaction", 25, 10001, {}, {"A1":1000,"A1_b":1000} )
s.set_param("k0", 100)
s.set_param("k1", k1)
s.set_param("d", 1)
s.add_reaction("k0", {"A1":1})
s.add_reaction("k0", {"A1_b":1})
s.add_reaction("k1*A1", {"A2":1, "A1":-1})
#s.add_reaction("d*A1", {"A1":-1})
#s.add_reaction("d*A2", {"A2":-1})

k_sum = 1/k1

for i in range(3):
    par = "k"+str(i+2)
    sp = "A" + str(i+3)
    sp_pre = "A" + str(i+2)
    s.add_reaction(par + "*" + sp_pre, {sp_pre:-1, sp:1})
    par_v = 1+0.1*i
    s.set_param(par, par_v)
    k_sum += 1/par_v
#    s.add_reaction("d*" + sp, {sp:-1})

s.set_param("k_sum",1/ k_sum)
s.add_reaction("k_sum * A1_b", {"A1_b":-1, "A_end":1})
s.add_reaction("d*A_end", {"A_end":-1})
s.add_reaction("d*" + sp, {sp:-1})

s.simulate()
s.plot_course(products=["A_end", sp])


#### Parallel ORDER reactions

k1 = 1
a = 2
b = 0.1
c = 0.3
d = 4.2
s = bs.SimParam("TestChainReaction2", 55, 10001, {}, {"A1":10,"A1_b":10} )
s.set_param("k0", 100)
s.set_param("k1", k1)
s.set_param("k1_a", a)
s.set_param("k1_b", b)
s.set_param("k2_a", c)
s.set_param("k2_b", d)
s.set_param("d", 1)
s.add_reaction("k0", {"A1":1})
s.add_reaction("k0", {"A1_b":1})

s.add_reaction("k1_a*A1", {"A2_pre_a":1, "A1":-1})
s.add_reaction("k1_b*A1", {"A2_pre_b":1, "A1":-1})
s.add_reaction("k2_a*A2_pre_a", {"A2":1, "A2_pre_a":-1} )
s.add_reaction("k2_b*A2_pre_b", {"A2":1, "A2_pre_b":-1} )

#k_sum = 1/(1/(a + b) + 2/(c + d)) # only if a = d and b = c
#k_sum = 2/(1/k1_a + 1/k1_b + 1/k2_a + 1/k2_b)
k_sum = 1/(1/a + 1/c) + 1/(1/b + 1/d)
s.set_param("k_sum", k_sum)
s.add_reaction("A1_b*k_sum", {"A1_b":-1, "A2_b":1})
s.add_reaction("d*A2_b", {"A2_b":-1})
s.add_reaction("d*A2", {"A2":-1})


#s.add_reaction("d*A1", {"A1":-1})
#s.add_reaction("d*A2", {"A2":-1})

print(s.compile_system())
s.simulate()
s.plot_course(products=["A2","A2_b"])

