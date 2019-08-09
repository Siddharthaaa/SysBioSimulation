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
    for i in range(150):
        s = SimParam("Lotka Voltera %d" %i, 50, 300,
                      {"k1":1, "k2":0.007, "k3":0.6 },
                      {"Prey":50, "Predator":200})
        s.add_reaction("k1*Prey",{"Prey":1})
        s.add_reaction("k2*Prey*Predator",{"Prey":-1, "Predator":1})
        s.add_reaction("k3*(Predator)",{"Predator":-1})
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
        
    ax.plot(0,0, lw = 0.5, color = "blue", label="Prey stoch"  )
    ax.plot(0,0, lw = 0.5, color = "orange", label="Predator stoch"  )
    
    res = np.array(res)
    res_prey = np.array(res_prey)
    res_predator = np.array(res_predator)
    
    
    
    ode_res = s.results["ODE"]
    ax.plot(tt,ode_res[:,1], color = "blue", lw = 4, label = "Prey (ODE)")
    ax.plot(tt,ode_res[:,2], color = "orange", lw = 4, label = "Predator (ODE)")
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
    t,p = sp.stats.ttest_ind(ode_res[:,2], res_predator.flatten(), equal_var=True)
    
    ax.legend(ncol=3)
    ax.set_title("Lotka Voltera")
    ax.set_xlabel("time")
    ax.set_ylabel("#")
    ax.set_ylim(0,500)
                  
    ret = s.compile_system()
    print(ret)
    simulate(s)
    s.plot()