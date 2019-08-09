# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:32:24 2019

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
    for i in range(80):
        s = SimParam("Circadian oscillation", 150,1000,
                      {"k1":1,"k3":1,"k5":1,"k2":0.1,"k4":0.2,"k6":0.2,"K_i":1, "n":10},
                      
                      {"RNA":0, "protein":1, "protein_p":4})
        s.add_reaction("k1*K_i**n/(K_i**n+protein_p**n)",{"RNA":1})
        s.add_reaction("k2*RNA",{"RNA":-1})
        s.add_reaction("k3*RNA",{"protein":1})
        s.add_reaction("k4*protein",{"protein":-1})
        s.add_reaction("k5*protein",{"protein_p":1})
        s.add_reaction("k6*protein_p",{"protein_p":-1})
        sims.append(s)
#        simulate(s)
    
    
    run_sims(sims,4)
    
    s.simulate_ODE = True
    s.simulate()
    res = []
    res_prey = []
    res_predator =[]
    d_prey_t = []
    d_pred_t = []
    ttt = []
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    for s in sims:
        r = s.results["stoch_rastr"]
        res.append(r)
        tt = r[:,0]
        prey = r[:,1]
        pred = r[:,2]
        ax.plot(tt,prey, lw = 0.1, color = "blue"  )
        ax.plot(tt,pred, lw = 0.1, color = "orange" )
        if all(pred*prey):
            res_prey.append(prey)
            res_predator.append(pred)
        else:
            if not all(prey):
                d_prey_t.append(tt[np.where(prey==0)[0][0]])
            if not all(pred):
                d_pred_t.append(tt[np.where(pred==0)[0][0]])
            
        
        
        #ax.plot(tt,ode_res[:,3], color = "green", lw = 0.1, label = name + "(ODE)")
    res = np.array(res)
    res_prey = np.array(res_prey)
    
    ode_res = s.results["ODE"]
    ax.plot(tt,ode_res[:,1], color = "blue", lw = 4, label = "Prey")
    ax.plot(tt,ode_res[:,2], color = "orange", lw = 4, label = "Predator")
    ax.plot(d_prey_t,np.ones(len(d_prey_t))*10 + 30*np.random.random(len(d_prey_t)), "x", color = "red", ms = 8, label = "Prey extinction")
    ax.plot(d_pred_t,np.ones(len(d_pred_t))*10 + 30*np.random.random(len(d_pred_t)), "o", color = "black", ms = 8, label = "Predator extinction")
    
    mean_prey = np.mean(res_prey,axis=0)
    mean_pred = np.mean(res_predator,axis=0)
    mean_p1 = np.mean(mean_prey)
    mean_p2 = np.mean(mean_pred)
    mean_p1_ode = np.mean(ode_res[:,1])
    ax.plot(tt,mean_prey,"--", color = "black", lw = 2, label= "mean bt. sims" )
    ax.plot(tt,mean_pred,"--", color = "black", lw = 2)
    ax.plot([tt[0], tt[-1]], [mean_p1, mean_p1],"-", lw=1, color="green", label ="total mean")
    ax.plot([tt[0], tt[-1]], [mean_p2, mean_p2],"-", lw=1, color="green")
#    ax.plot([tt[0], tt[-1]], [mean_p1_ode, mean_p1_ode],"-", lw=1, color="red")
    
    t,p = sp.stats.ttest_ind(ode_res[:,1], res_prey.flatten(), equal_var=True)
    
    ax.legend()
    ax.set_title("Circadian oscillation")
    ax.set_xlabel("time")
    ax.set_ylabel("#")
    ax.set_ylim(0,500)
                  
    ret = s.compile_system()
    print(ret)
    simulate(s)
    s.plot()
