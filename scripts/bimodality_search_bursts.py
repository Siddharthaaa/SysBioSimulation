
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:14:15 2019

@author: Timur
"""
from bioch_sim import *

s2=2
d2 = 0.15

intensity_th = 5


def plot_3d_heat(sims, axes = ("s2", "d2"), fig = None, ignore_extremes = False,
                 recognize_threshold = 0):
    if fig is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        ax = fig.add_subplot(1, 2, 2, projection='3d')
    
    x = np.zeros((len(sims), len(sims[0])))
    y = np.copy(x)
    z = np.copy(x)
   
    i = 0
    for ss in sims:
        j=0
        for s in ss:
#            x[i,j] = s.params[axes[0]]
#            y[i,j] = s.params[axes[1]]
            x[i,j] = s.params[axes[0]]
            y[i,j] = s.params[axes[1]]
            z[i,j] = s.get_bimodality(ignore_extremes=ignore_extremes,
             recognize_threshold = recognize_threshold)
            
            
            j+=1
        i+=1
    
    #ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    ax.plot_surface(x,y,z,  cmap=cm.coolwarm,  alpha = 0.7 )
    ax.plot_wireframe(x,y,z,  color="black", lw=0.3)
#    ax.legend()
    #ax.plot_wireframe(x,y,z,  cmap=cm.coolwarm)
#    ax.set_xlabel("\n" + axes[0],linespacing=2.2)
#    ax.set_ylabel("\n" + axes[1],linespacing=2.2)
    ax.set_xlabel("\n" + axes[0],linespacing=2.2)
    ax.set_ylabel("\n" + axes[1],linespacing=2.2)
    ax.set_zlabel("\nBscore", linespacing=2.2)
    #ax.set_zlim(0)
    plt.show()
    return ax
    
def plot_heatmap( sims, axes = ("s2", "d2"), ignore_extremes = False, recognize_threshold = 0) :
    
    sims = np.array(sims)
    ax1, ax2 = axes
    
    tck_c_y = 6
    tck_c_x = 4
    
    y_labels = ["%2.2f" % s.params[ax1] for s in  sims[:,1]]
    x_labels = ["%2.2f" % s.params[ax2] for s in sims[1,:]]
    
    shape = sims.shape
    b_scores = np.zeros(shape)
    
    for i in range(0,shape[0]):
        for j in range(0, shape[1]):
            b_scores[i,j] =  sims[i,j].get_bimodality(ignore_extremes=ignore_extremes,
                    recognize_threshold = recognize_threshold)

#    heatmap

#    names = (s.name for s in sims)
    
    step_y = int(len(y_labels)/tck_c_y)
    step_x = int(len(x_labels)/tck_c_x)
    x_ticks = np.arange(0,stop = len(x_labels), step=step_x, dtype = np.int)
    y_ticks = np.arange(0,stop = len(y_labels), step=step_y, dtype = np.int)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
#    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels([y_labels[i] for i in y_ticks])
    ax.set_xticklabels([x_labels[i] for i in x_ticks])
    im = ax.imshow(b_scores, aspect="auto")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Bscore", rotation=-90, va="bottom")
    title = "Bscore Search"
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ax1)
    ax.set_xlabel(ax2)
    return ax


if __name__ == "__main__":
    
    flattened_sims = []
    shape = (30,30)
    
    sims = []
    
    #create simulations
    for i in range(0,shape[0]):
        sims.append([])
        k_syn = 40
        k_on = 0.05
        k_off =0.2
        d1 = 0.1
        s2 = 1 + i*0.5
        for j in range(0, shape[1]):
            
            s1=2
#            s2 = 1
            s3 = 0.1
            d0 = 0.1
            d2 = 0.01 + j*0.02
            d3=1
#            incl_ss = 10
#            skip_ss = 5
#            ret_ss = 1
#
#            psi = incl_ss/(incl_ss+skip_ss) 
            
            #psi = incl_ss/(incl_ss+skip_ss)
            
            name="s2=%2.2f,d2 =%2.2f" % (s2, d2)
            s = SimParam(name,5000, 10000,
                         params = {"k_on": k_on, "k_off": k_off, "v_syn": k_syn, "s1": s1,
                                   "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
                         init_state = {"Pr_on": 0, "Pr_off": 1, "pre_RNA": 0,
                                       "Incl": 0, "Skip": 0, "ret": 0})
            
            s.simulate_ODE = True
            
            s.add_reaction("k_on*Pr_off", {"Pr_on":1, "Pr_off":-1})
            s.add_reaction("k_off*Pr_on", {"Pr_off":1, "Pr_on":-1})
            s.add_reaction("v_syn*Pr_on", {"pre_RNA":1} )
            s.add_reaction("d0*pre_RNA", {"pre_RNA":-1} )
            s.add_reaction("s1*pre_RNA", {"pre_RNA":-1, "Incl":1})
            s.add_reaction("d1*Incl", {"Incl": -1}  )
            s.add_reaction("s2*pre_RNA" ,  {"pre_RNA":-1, "Skip":1})
            s.add_reaction("d2*Skip", {"Skip":-1}  )
            s.add_reaction("s3*pre_RNA", {"pre_RNA":-1, "ret":1} )
            s.add_reaction("d3*ret",  {"ret": -1} )
            sims[i].append(s)
            flattened_sims.append(s)
            
            print("Created sim: %s" % s.param_str())
        
    
    run_sims(flattened_sims, 4)
    
    res = []  
    lengths = []
    for s in flattened_sims:
        
        b_score = s.get_bimodality(ignore_extremes=False, ignore_fraction=0, recognize_threshold=intensity_th)
        l = len(s.results["PSI"][1])
        lengths.append(l)
        res.append(b_score)
    res = np.array(res)
    
#    s.compute_psi()
    
    indices= np.argsort(-res) 
    
    best_sim = flattened_sims[indices[1]]
    
    best_sim.colors = ["red","black", "blue", "green", "red", "black"]
    best_sim.compute_psi(ignore_fraction=0, recognize_threshold=intensity_th)
    plot_hist(best_sim.results["PSI"][1])
    ax1 = best_sim.plot_course(products=["Skip", "Incl"], rng = 300)
    ax1.legend(ncol=2, fontsize = 10)
    ax1_psi = ax1.twinx()
    end_ind = np.where(best_sim.results["PSI"][0] < 150)
    ax1_psi.plot(best_sim.results["PSI"][0][end_ind], best_sim.results["PSI"][1][end_ind], "r.", color="blue", label = "PSI")
    ax1_psi.legend()
    ax1_psi.set_ylim(0,1.5)
    ax1_psi.set_ylabel("PSI")
    ax2 = best_sim.plot_course(products=["Pr_on", "pre_RNA"], rng = 300)
    ax2.legend(ncol=2, fontsize = 10)
    
#    flattened_sims[indices[0]].simulate_ODE = True
#    flattened_sims[indices[0]].simulate()
#    flattened_sims[indices[0]].get_bimodality()
    
    plot_3d_heat(sims, ignore_extremes=False, recognize_threshold = intensity_th)
    plot_heatmap(sims, ignore_extremes=False, recognize_threshold = intensity_th)
    
    
