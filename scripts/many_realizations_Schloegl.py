# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:35:51 2019

@author: Timur
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:14:15 2019

@author: Timur
"""
from bioch_sim import *
if __name__ == "__main__":
    
    
    sims = []
    for i in range(5):
        s = SimParam("Schlögl system",100 ,1000,
                     {"c1":3e-7 , "c2": 1e-4, "c3": 1.0e-3, "c4":3.5 },
                      {"A": 1e5, "B": 2e5, "X": 200+i*0})
        
        s.add_reaction("c2*X*X*X", {"X":-1, "A":1} )
        s.add_reaction("c1*X*X*A", {"X":1, "A":-1} )
        s.add_reaction("c3*B", {"X":1, "B":-1} )
        s.add_reaction("c4*X", {"X":-1, "B":1} )
        s.simulate_ODE = True
        s.simulate()
        s.colors = ["red","orange","green"]
        s.plot(products=["X"])
        s.plot_course(products=["X"])
        
        sims.append(s)
#        simulate(s)
    
    
    sims[0].simulate_ODE = True
    run_sims(sims,4)
    
    res = []  
    
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    
    for s in sims:
        r = s.results["stoch_rastr"]
        tt = r[:,0]
        result = s.get_res_col("X")
        ax.plot(tt,result, lw = 0.1, color = "grey" )
        res.append(result)
       
    
    ode_res = sims[0].results["ODE"]
#    ax.plot(tt,ode_res[:,3], color = "green", lw = 0.1, label = name + "(ODE)")
    res = np.array(res)
    mean_psi = np.mean(res,axis=0)
   
    ax.plot(tt,sims[0].get_res_col("X", "ODE"), color = "green", lw = 5, label = "X")
   
    ax.legend()
    #fig.suptitle("200 Realisationen")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("#")
    ax.set_title("Einzelne Spezies")
    
    ret = s.compile_system()
    print(ret)
    s.simulate()
    s.plot()

