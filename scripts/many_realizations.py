# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:14:15 2019

@author: Timur
"""
from bioch_sim import *
if __name__ == "__main__":
    
    v_syn = 10
    s1 = 0.3
    s2= 0.3
    s3=1
    d0 = 0.1
    d1 = 0.02
    d2= 0.03
    d3= 1
    
    pre_ss  = v_syn/(d0+s1+s2+s3)
    incl_ss = pre_ss*s1/d1
    skip_ss = pre_ss*s2/d2
    ret_ss = pre_ss*s3/d3
    psi_ss = incl_ss/(incl_ss+skip_ss)
    
    sims = []
    for i in range(2000):
        s = SimParam("Steady State %d" % i,500 ,1000,
                     {"v_syn": v_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0,
                      "d1": d1, "d2": d2, "d3": d3},
#                      {"pre_RNA": int(pre_ss), "Incl": int(incl_ss),
#                       "Skip": int(skip_ss), "ret": int(ret_ss)})
                      {"pre_RNA": 0, "Incl": 0,
                       "Skip": 0, "ret": 0})
        
        s.add_reaction("v_syn", {"pre_RNA":1} )
        s.add_reaction("d0*pre_RNA", {"pre_RNA":-1} )
        s.add_reaction("s1*pre_RNA", {"pre_RNA":-1, "Incl":1})
        s.add_reaction("d1*Incl", {"Incl": -1} )
        s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1})
        s.add_reaction("d2*Skip", {"Skip":-1}  )
        s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1} )
        s.add_reaction("d3*ret",  {"ret": -1} )
        s.colors = ["green", "blue", "orange", "red" ]
#        s.simulate(ODE=True)
#        s.plot(products=["Incl","Skip"], psi_hist=True)
        
        sims.append(s)
#        simulate(s)
    
    
    sims[0].simulate_ODE = True
    run_sims(sims,4)
    
    res = []  
    res_incl =[]
    res_skip =[]
    
    fig, ax_arr = plt.subplots(1,2, figsize=(2*7,7))
    ax = ax_arr[0]
    ax_psi = ax_arr[1]
    for s in sims:
        r = s.results["stoch_rastr"]
        psi = r[:,2]/(r[:,2]+r[:,3])
        np.nan_to_num(psi, False)
        res.append(psi)
        
        tt = r[:,0]
        ax.plot(tt,r[:,2], lw = 0.1, color = "grey" )
        ax.plot(tt,r[:,3], lw = 0.1, color = "grey" )
        
        res_incl.append(r[:,2])
        res_skip.append(r[:,3])
        
        ax_psi.plot(tt,psi, lw = 0.1, color = "grey" )
    
    res_incl = np.array(res_incl)    
    res = np.array(res)    
    res_skip = np.array(res_skip)    
    p_val_psi = sp.stats.ttest_1samp(res[:,-1], psi_ss)
    p_val_incl = sp.stats.ttest_1samp(res_incl[:,-1], incl_ss)
    p_val_skip = sp.stats.ttest_1samp(res_skip[:,-1], skip_ss)
    
    ode_res = sims[0].results["ODE"]
#    ax.plot(tt,ode_res[:,3], color = "green", lw = 0.1, label = name + "(ODE)")
    res = np.array(res)
    mean_psi = np.mean(res,axis=0)
   
    ax.plot([tt[0],tt[-1]],[incl_ss,incl_ss],"--", color = "red", lw = 1, label = "steady state")
    ax.plot([tt[0],tt[-1]],[skip_ss,skip_ss],"--", color = "green", lw = 1)
    ax.plot(tt,ode_res[:,2], color = "red", lw = 5, label = "Incl")
    ax.plot(tt,ode_res[:,3], color = "green", lw = 5, label = "Skip")
    ax.plot(tt,np.mean(res_incl,axis=0),"-", color = "black", lw = 2, label= "mean" )
    ax.plot(tt,np.mean(res_skip,axis=0),"-", color = "black", lw = 2)
   
    
   
    
    ax.legend()
    #fig.suptitle("200 Realisationen")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("#")
    ax.set_title("Einzelne Spezies")
                 
    ax_psi.plot(tt,mean_psi, color = "green", lw = 3, label = "mean(PSI)")
    ax_psi.plot([tt[0],tt[-1]],[psi_ss,psi_ss],"--", color = "red", lw = 1, label = "PSI steady state")
    ax_psi.set_ylabel("PSI = Incl/(Incl + Skip)")
    ax_psi.set_xlabel("Zeit [s]")
    ax_psi.legend()
    
    ax_psi.set_title("Verlauf von PSI")
                  
    
    ret = s.compile_system()
    print(ret)
    s.simulate()
    s.plot()
    

