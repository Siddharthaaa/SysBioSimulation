# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:35:37 2019

@author: Timur
"""

from bioch_sim import *
import time
import os
if __name__ == '__main__':
    print(os.getcwd())
    start = time.time()
    sims = []
    flattened_sims = []
    shape = (6,30)
    
    #create simulations
    for i in range(0,shape[0]):
        sims.append([])
        v_syn = 4*(1+i)
        
        for j in range(0, shape[1]):
            
            s1=j+0
            s2 = shape[1]-s1-1
            s3 = 1
            psi = 0.5
            incl_ss = 10
            skip_ss = 5
            ret_ss = 3
            d0 = 0.1
            d1 = d2= 2
            
            d3=0.1
            pre_ss = v_syn/(s1+s2+s3+d0)
            incl_ss=pre_ss*s1/d1
            skip_ss=pre_ss*s2/d2
            ret_ss=pre_ss*s3/d3
            
#            
#            d1=s1*pre_ss/incl_ss
#            d2=s2*pre_ss/skip_ss
#            d3 = s3*pre_ss/ret_ss
#            
            
            psi = incl_ss/(incl_ss+skip_ss) 
            
            #psi = incl_ss/(incl_ss+skip_ss)
            
            name="v_syn=%2.2f, s1=%2.2f (psi:%1.2f, I:%2.2f, S:%2.2f)" % (v_syn, s1, psi, incl_ss, skip_ss)
            s = SimParam(name,10000, 20000,
                         params = {"v_syn": v_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
                         init_state = {"pre_RNA": int(pre_ss), "Incl": int(incl_ss),
                                          "Skip": int(skip_ss), "ret": int(ret_ss)})
            
            s.simulate_ODE = False
            
            s.add_reaction("v_syn", {"pre_RNA":1} )
            s.add_reaction("d0*pre_RNA", {"pre_RNA":-1} )
            s.add_reaction("s1*pre_RNA", {"pre_RNA":-1, "Incl":1})
            s.add_reaction("d1*Incl", {"Incl": -1}  )
            s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1})
            s.add_reaction("d2*Skip", {"Skip":-1}  )
            s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1} )
            s.add_reaction("d3*ret",  {"ret": -1} )
            sims[i].append(s)
            flattened_sims.append(s)
            s.expected_psi = psi
            
            print("Created sim: %s" % s.param_str())
        
    
    run_sims(flattened_sims,4)
        
    end = time.time()
    print("runtime: %f s" % (end - start))
    
    incl_sds = np.zeros(shape)
    sd_psi = np.zeros(shape)
    mean_psi = np.zeros(shape)
    sims = np.array(sims)
    for i in range(0,shape[0]):
        for j in range(0, shape[1]):
             incl_sds[i,j] = np.std(sims[i,j].get_res_col("Incl")[1000:])
             psi = sims[i,j].compute_psi()
             sd_psi[i,j] = np.std(psi)
             mean_psi[i,j] = np.mean(psi)
    
    y_labels = [s.params["v_syn"] for s in  sims[:,1]]
    x_labels = ["%2.2f" % s.params["s1"] for s in sims[1,:]]        
    colors = cm.rainbow(np.linspace(0, 1, len(sims)))
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    i=0
    for v_syn_sim in sims:
        name = "v_syn: %s" % v_syn_sim[0].params["v_syn"]
        psis = [s.compute_psi() for s in v_syn_sim]
        mean_psis = np.mean(psis,1)
        sd_psis = np.std(psis,1)
        ax.plot(mean_psis, sd_psis, ms=8, color= colors[i], label = name) 
        i+=1
    ax.legend()
    ax.set_ylabel("SD(PSI)")
    ax.set_xlabel("mean(PSI)")
    ax.set_title("High degradation rate (d1=d2=%.2f)" %d1)
    
    
    
# 