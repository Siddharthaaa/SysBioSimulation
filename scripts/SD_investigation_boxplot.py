# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:48:36 2019

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
    shape = (5,30)
    
    #create simulations
    for i in range(0,shape[0]):
        sims.append([])
        v_syn = 6*(1+i)
        
        for j in range(0, shape[1]):
            
            s1=j+0
            s2 = shape[1]-s1-1
            s3 = 1
            psi = 0.5
            incl_ss = 10
            skip_ss = 5
            ret_ss = 3
            d0 = 0.1
            d1 = d2= 0.1
            
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
            s = SimParam(name,10000, 10000,
                         params = {"v_syn": v_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
                         init_state = {"pre_RNA": int(pre_ss), "Incl": int(incl_ss),
                                          "Skip": int(skip_ss), "ret": int(ret_ss)})
                    
            s.states ={}
            s.states["Incl"] = incl_ss
            s.states["Skip"] = skip_ss
            
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
    fig, ax = plt.subplots(1,3, figsize=(5*3,5))
    ax_box = ax[1]
    ax_ranbow = ax[0]
    i=0
    names = []
    mean_ps = []
    psis_arr =[]
    for v_syn_sim in sims:
        skip_ss = v_syn_sim[0].states["Skip"]
        incl_ss = v_syn_sim[0].states["Incl"]
        
        name = "v_syn: %s, I+S: %d" % (v_syn_sim[0].params["v_syn"],incl_ss+skip_ss )
        names.append(name)
        psis = [s.compute_psi() for s in v_syn_sim]
        psis_arr.append(psis)
        mean_psis = np.mean(psis,1)
        mean_ps.append(mean_psis)
        sd_psis = np.std(psis,1)
        ax_ranbow.plot(mean_psis, sd_psis, ms=8, color= colors[i], label = name) 
        i+=1
    ax_ranbow.legend(prop = {"size":10})
    ax_ranbow.set_ylabel("SD(PSI)")
    ax_ranbow.set_xlabel("mean(PSI)")
    ax_ranbow.set_title("Low degr. rate (d1=d2=%.2f)" %d1)
    
    ind = 1
    for i in [0, len(names)-1]:
        
#        ax[ind].boxplot(psis_arr[i])
        ax[ind].violinplot(psis_arr[i])
        
        ax[ind].set_title(names[i])
        ax[ind].set_ylabel("PSI")
        ax[ind].set_xlabel("mean(PSI)")
        tck_c_x = 5
        step_x = int(len(mean_ps[i])/tck_c_x)
        x_ticks = np.arange(0,stop = len(mean_ps[i]), step=step_x, dtype = np.int)
        ax[ind].set_xticks(x_ticks+1)
        ax[ind].set_xticklabels("%2.2f" % m for m in mean_ps[i][x_ticks+1])
        ind+=1
    
    
# 