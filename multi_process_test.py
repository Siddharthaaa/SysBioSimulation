# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:44:52 2019

@author: imb30
"""

from bioch_sim import *

if __name__ == "__main__":
    p = []
    start = time.time()
    sims = []
    for i in range(24):
        p.append(i+1)
    
        s = SimParam("Test simulation",10000 ,10000,
                     {"v_syn": 50, "s1": 0.3, "s2": 0.3, "s3": 1, "d0": 0.1,
                      "d1": 0.02, "d2": 0.03, "d3": 1},
                      {"pre_RNA": 5, "Incl": 0, "Skip": 0, "ret": 0})
        
        s.add_reaction("v_syn", {"pre_RNA":1} )
        s.add_reaction("d0*pre_RNA", {"pre_RNA":-1} )
        s.add_reaction("s1*pre_RNA", {"pre_RNA":-1, "Incl":1})
        s.add_reaction("d1*Incl", {"Incl": -1}  )
        s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1})
        s.add_reaction("d2*Skip", {"Skip":-1}  )
        s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1} )
        s.add_reaction("d3*ret",  {"ret": -1} )
        sims.append(s)
        s.simulate()
    
    t1 = time.time()-start    
    
    tt = []
    pp = []
    for i in range(8):
        pp.append(i+1)
        start = time.time()
        run_sims(sims,i+1)
        tt.append(time.time() - start)
        
    fig, ax = plt.subplots(1,1, figsize=(7,7))

    ax.plot(pp, tt, color = "green", label="With childs", lw=2)
    ax.plot([pp[0],pp[-1]],[t1,t1], "--", color = "blue", label="Without childs", lw=2)
    ax.legend()
    ax.set_title("24 Simulations (4 CPUs + HT)")
    ax.set_xlabel("# Child procs")
    ax.set_ylabel("Runtime [s]")
    ax.set_ylim(0)