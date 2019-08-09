
# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pylab as pl
import numpy as np
import scipy as sp
import time
import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec
from scipy.optimize import fsolve
from scipy.integrate import odeint
# inline displaying
#%matplotlib inline
# settings for plots
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['font.family'] = ['sans-serif']
line_width = 2
color_pre = 'dodgerblue'
color_Incl = 'darkorange'
color_Skip = "green"
color_phase_space = 'dimgrey'
color_initial_state = 'crimson'
color_steady_state = 'saddlebrown'


start = time.time()

def solve_ODE_transcription_splicing(x,t,param):
   
    x_arr= x.tolist()
    x_arr.insert(0,1.)
    dx = param.dot(x_arr)
    #print(x)
    return dx
  
# analytical solution based on the equations above

def stoch_sim_transcription_splicing(param,initial_state,tf):
    k_syn,d0, s1, s2, s3, d1, d2, d3 = param[0],param[1],param[2],param[3],param[4],param[5],param[6],param[7]
    
    # reaction matrix, the systems state includes the time points of the reactions in the first column
    reactions = np.array([[0,1,0,0,0],
                         [0,-1,0,0,0],
                         [0,-1,1,0,0],
                         [0,0,-1,0,0],
                         [0,-1,0,1,0],
                         [0,0,0,-1,0],
                         [0,-1,0,0,1],
                         [0,0,0,0,-1]])
    
    # initialise the systems state
    state = np.zeros(5)
    state[1] = initial_state[0]
    state[2] = initial_state[1]
    state[3] = initial_state[2]
    state[4] = initial_state[3]
    STATE = state
    
    tt = 0
    while tt <= tf:
        # sample two random numbers uniformly between 0 and 1
        rr = sp.random.uniform(0,1,2)
        
        #gamma der exponetialverteilung?
        #a_0 = beta + gamma_m*state[1] + k*state[1] + gamma_p*state[2] 
        #a_0 = k_syn + (d0+s1+s2+s3)*state[1] + d1*state[2] + d2*state[3] + d3*state[4]
        #einzelne gammas im array
        #a_s = np.array([beta,gamma_m*state[1],k*state[1],gamma_p*state[2]],dtype = float)
        a_s = np.array([k_syn,
                        d0*state[1],
                        s1*state[1],
                        d1*state[2],
                        s2*state[1],
                        d2*state[3],
                        s3*state[1],
                        d3*state[4]],dtype = float)
        a_0 = a_s.sum()
        #print(a_0,a_s.sum())            
        # time step 
        tt = tt - 1. / a_0 * sp.log(1. - rr[0])
        state[0] = tt
        
        # find the next reaction
        prop = rr[1] * a_0
        cum_a_s = np.cumsum(a_s)
        #print(cum_a_s, prop)
        ind = np.where(prop <= cum_a_s)[0][0]
        
        # update the systems state
        state = state+reactions[ind]
        
        STATE = np.vstack((STATE,state))
    
    return STATE
# define parameters  

cell_count = 2

ode_results = []
stoch_sim_results = []
psi_mean_last_third =[]

for cell_nb in range(0,cell_count):
    k_syn, k_syn_sd = 8. , 2
    k_syn = np.random.normal(k_syn, k_syn_sd)
    d0 = 0.05
    s1, s1_sd = 0.6, 0.3
    s1 = np.random.normal(s1,s1_sd)
    s2 = 0.3
    s3 = 0.1
    d1 = 0.1
    d2 = 0.1
    d3 = 0.1
    
    param = [k_syn,d0,s1,s2,s3,d1,d2,d3]
    
    # calculate steady states
    #steady_state_RNA = k_1/d_1
    #steady_state_protein = k_2*k_1/d_1/d_2
    
    steady_state_pre = k_syn/(d0+s1+s2+s3)
    steady_state_Incl = steady_state_pre*s1/d1
    steady_state_Skip = steady_state_pre*s2/d2
    
    # define intial conditions
    #RNA_0 = 0
    #protein_0 = 0
    #initial_state = [RNA_0,protein_0]
    
    pre_0 = 0
    Incl_0 = 0
    Skip_0 = 0
    ret_0 = 0 
    
    initial_state = [pre_0,Incl_0,Skip_0,ret_0]
    
    # simulation time
    tf = 200
    
    sim = stoch_sim_transcription_splicing(param,initial_state,tf)
    stoch_sim_results.append(sim)
    
    #analyze the last third
    start_ind = int(-len(sim)/3)
    incl_counts = sim[start_ind:,2]
    skip_counts = sim[start_ind:,3]
    
    psi = incl_counts/(incl_counts+skip_counts)
    psi_mean_last_third.append(np.mean(psi))
    
    #determinisitic simulation
    tt = sp.linspace(0,tf,200)
    y_0 = initial_state
    
    m = np.array([[k_syn, -(d0+s1+s2+s3),0,0,0],
                  [0, s1,-d1,0,0],
                  [0, s2,0,-d2,0],
                  [0, s3,0,0,-d3]])
    
    sol_deterministic = odeint(solve_ODE_transcription_splicing,y_0,tt,args = (m,))
    ode_results.append(sol_deterministic)



fig,ax = pl.subplots(2,2, figsize = (14,15))
ax[0,0].plot(sim[:,0],sim[:,1], label = 'pre-RNA', color = color_pre, lw = line_width,drawstyle = 'steps')
ax[0,0].plot(tt,sol_deterministic[:,0], color = color_pre, lw = 2*line_width, label = 'pre-RNA, ODE solution')
ax[0,0].plot(tt,sol_deterministic[:,1], color = color_Incl, lw = 2*line_width, label = 'Incl, ODE solution')
ax[0,0].plot(tt,sol_deterministic[:,2], color = color_Skip, lw = 2*line_width, label = 'Skip, ODE solution')
ax[0,0].plot(sim[:,0],sim[:,2], label = 'Incl', color = color_Incl,lw = line_width,drawstyle = 'steps')
ax[0,0].plot(sim[:,0],sim[:,3], label = 'Skip', color = color_Skip,lw = line_width,drawstyle = 'steps')
#ax[0].plot(tt,sol_deterministic[:,1], color = color_Incl, lw = 2*line_width, label = 'Protein, ODE solution')
ax[0,0].plot(sim[:,0],np.zeros(len(sim[:,0])) + steady_state_pre,'--', label = 'steady state pre-RNA',
          color = color_pre)
ax[0,0].plot(sim[:,0],np.zeros(len(sim[:,0])) + steady_state_Incl,'--', label = 'steady state Incl',
          color = color_Incl)
ax[0,0].plot(sim[:,0],np.zeros(len(sim[:,0])) + steady_state_Skip,'--', label = 'steady state Skip',
          color = color_Skip)
ax[0,0].legend(loc = 'best')
ax[0,0].set_xlabel('Time')
ax[0,0].set_ylabel('Molecule number')
#ax[0,0].set_ylim(0,2*steady_state_pre)
ax[0,0].set_xlim(0,210)

ax[0,1].plot(sim[:,1],sim[:,2], label = 'state trajectory', lw = line_width,color = color_phase_space)
ax[0,1].plot(steady_state_pre,steady_state_Incl,'o',ms = 15,label = 'steady state', color = color_steady_state)
ax[0,1].plot(pre_0,Incl_0,'o',ms = 15,label = 'Initial state', color = color_initial_state)
ax[0,1].legend(loc = 'best')
ax[0,1].set_ylabel('Incl')
ax[0,1].set_xlabel('pre')
ax[0,1].set_title('Phase space')

ax[1,1].hist(psi_mean_last_third, label="PSI")
ax[1,1].set_xlabel("psi=Incl/(Incl+Skip)")

ax[1,0].hist(sim[:,(1,2,3)], label=("pre","Incl","Skip"))


end = time.time()
print(end - start)