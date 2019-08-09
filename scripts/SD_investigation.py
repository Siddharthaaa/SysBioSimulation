# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:26:39 2019

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
    shape = (30,30)
    
    #create simulations
    for i in range(0,shape[0]):
        sims.append([])
        k_syn = 1*(1+i)
        
        for j in range(0, shape[1]):
            
            s1=0.1*(j+1)
            s2 = 1
            s3 = 1
            psi = 0.5
            incl_ss = 10
            skip_ss = 5
            ret_ss = 1
#            s1 = 1
#            s2 = 1
#            s3 = 0.1
            d0 = 0.1
#    
            pre_ss = k_syn/(s1+s2+s3+d0)
#            
            d1=s1*pre_ss/incl_ss
            d2=s2*pre_ss/skip_ss
            d3 = s3*pre_ss/ret_ss
#            
            
            psi = incl_ss/(incl_ss+skip_ss) 
            
            #psi = incl_ss/(incl_ss+skip_ss)
            
            name="v_syn=%2.2f, s1=%2.2f (psi:%1.2f, I:%2.2f, S:%2.2f)" % (k_syn, s1, psi, incl_ss, skip_ss)
            s = SimParam(name,3000, 6000,
                         params = {"v_syn": k_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
                         init_state = {"pre_RNA": 0, "Incl": 0, "Skip": 0, "ret": 0})
            
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
        
#    for sim in flattened_sims:
#        #analyze the last third
#        sim_st_raster = sim.results["stoch_rastr"]
#        start_ind = int(len(sim_st_raster)*1/3)
#        incl_counts = sim_st_raster[start_ind:,2]
#        skip_counts = sim_st_raster[start_ind:,3]
#        
#        psi = incl_counts/(incl_counts+skip_counts)
#        np.nan_to_num(psi, False)
#        sim.results["PSI"] = psi
#    psi=plot_psi_to_cv(sims)
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

#    heatmap

#    names = (s.name for s in sims)
    tck_c_y = 10
    tck_c_x = 6
    step_y = int(len(y_labels)/tck_c_y)
    step_x = int(len(x_labels)/tck_c_x)
    x_ticks = np.arange(0,stop = len(x_labels), step=step_x, dtype = np.int)
    y_ticks = np.arange(0,stop = len(y_labels), step=step_y, dtype = np.int)
    fig, ax = plt.subplots()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
#    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels([y_labels[i] for i in y_ticks])
    ax.set_xticklabels([x_labels[i] for i in x_ticks])
    im = ax.imshow(sd_psi, aspect="auto", )
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("SD(PSI)", rotation=-90, va="bottom")
    ax.set_title("Incl:10, Skip:5", fontsize=15)
    ax.set_ylabel("v_syn")
    ax.set_xlabel("s1")

#fig.savefig("heat_map" + ".png", format="png",dpi=250)
    