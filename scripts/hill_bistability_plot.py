# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:34:19 2019

@author: imb30
"""
from bioch_sim import *

k_on=0.050
k_off=0.000
v_syn=10.000
s1=0.500
Ka1=81.000
n1=6.000
s2=1.000
s3=0.100
d0=0.100
d1=0.100
d2=0.100
d3=0.500
s1_t=14.000


fig = plt.figure()
ax = fig.add_subplot(1,1,1)

count = 4
colors = cm.rainbow(np.linspace(0, 1, count))

for i in range(count):
    Incl_init =  50+ i 
    name="Ka1=%2.2f,s1_t =%2.2f, n1=%2.2f" % (Ka1, s1_t, n1)
    s = SimParam(name, 120, 1000,
                 params = {"k_on": k_on, "k_off": k_off, "v_syn": v_syn, "s1": s1, "Ka1":Ka1, "n1":n1,
                           "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3, "s1_t":s1_t},
                 init_state = {"Pr_on": 1, "Pr_off": 0, "pre_RNA": 0,
                               "Incl": Incl_init, "Skip": 0, "ret": 0})
    
    s.simulate_ODE = True
    
    s.add_reaction("k_on*Pr_off", {"Pr_on":1, "Pr_off":-1})
    s.add_reaction("k_off*Pr_on", {"Pr_off":1, "Pr_on":-1})
    s.add_reaction("v_syn*Pr_on", {"pre_RNA":1} )
    s.add_reaction("d0*pre_RNA", {"pre_RNA":-1} )
    s.add_reaction("s1*pre_RNA + pre_RNA* s1_t * hill(Incl, Ka1, n1)", {"pre_RNA":-1, "Incl":1})
    s.add_reaction("d1*Incl", {"Incl": -1}  )
    s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1})
    #            s.add_reaction("s2*pre_RNA + 5* (Skip > 0)* 1/(1+(Ka2/Skip)**n2)" ,  {"pre_RNA":-1, "Skip":1})
    s.add_reaction("d2*Skip", {"Skip":-1}  )
    s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1} )
    s.add_reaction("d3*ret",  {"ret": -1} )
    s.simulate()
    
    t = s.results["ODE"][:,0]
    res = s.get_res_col("Incl", method = "ODE")
    
    ax.plot(t, res, label = "Incl_init: %d" % Incl_init, c=colors[i])
    res = s.get_res_col("Skip", method = "ODE")
    ax.plot(t, res,"--", label = "Incl_init: %d" % Incl_init, c=colors[i])

custom_lines = [plt.Line2D([0], [0], color="black", label ="Incl"),
                plt.Line2D([0], [0], linestyle= "--", color="black", label ="Skip")]
ax.legend(custom_lines, ["Incl", "Skip"])
ax.set_title("Bistability by variation of init. Incl", fontsize=12)
