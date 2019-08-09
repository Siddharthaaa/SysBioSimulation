
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:14:15 2019

@author: Timur
"""
from bioch_sim import *
if __name__ == "__main__":
    
    
    k_1 = 8e6
    k_2 = 1.33e5
    k_3 = 667
    k4 = 1e-2
    k1 = k_1*k4
    k2 = k_2*k4
    k3 = k_3*k4
    
    
    sims = []
    for i in range(10):
        s = SimParam("Schlögl system",1 ,100000,
                     {"k1":k1 , "k2": k2, "k3": k3, "k4":k4},
                      {"X":200+10*i })
        
        s.add_reaction("k1", {"X":1} )
        s.add_reaction("k2*X", {"X":-1} )
        s.add_reaction("k3*X*X", {"X":1} )
        s.add_reaction("k4*X*X*X", {"X":-1} )
        s.simulate_ODE = True
#        s.simulate(max_steps=1e9)
#        s.colors = ["red","orange","blue"]
#        s.plot(products=["X"])
#        s.plot_course(products=["X"])
        
        sims.append(s)
#        simulate(s)
    
    
    sims[0].simulate_ODE = True
    run_sims(sims,4)
    
    res = []  
    
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    
    bscores =np.zeros(len(sims))
    i=0
    for s in sims:
        r = s.results["stoch_rastr"]
        tt = r[:,0]
        result = s.get_res_col("X")
        ax.plot(tt,result, lw = 0.1, color = "grey" )
        res.append(result)
        bscores[i] = (get_multimodal_scores(result)[0])[:,0].max()
        i+=1
    
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
   
    indices = np.argsort(bscores)
    sims[indices[-1]].plot()
    plot_hist(sims[indices[-1]].get_res_col("X"))
    for i in indices:
        s = sims[i]
        result = s.get_res_col("X")
        plot_hist(result)
#    plot_hist(result)
    
#    ret = s.compile_system()
#    print(ret)
#    s.simulate()
#    s.plot()

