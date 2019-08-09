

import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import scipy.stats as st
import time
import math
from threading import Thread
from multiprocessing import Process, Manager, Queue
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import gridspec
from scipy.optimize import fsolve
from scipy.integrate import odeint
import pickle
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


thread_count = 2

class SimParam():
    def __init__(self, name, time=200, params={}, init_state=[]):
        self.name = str(name)
        self.runtime = time
        self.raster =  sp.linspace(0,time,int(time+1))
        self.params=params
        self.init_state=init_state
        self.results={}
        self.init_stat = {}
        self.id=id(self)
    def set_param(self, name, value):
        self.params[name] = value
    def get_all_params(self):
        params = {}
        params["constants"] = np.array(list(self.params.values()))
        params["init_state"] = np.array(list(self.init_state.values()))
        params["raster"] = np.array(self.raster)
        params["runtime"] = self.runtime
        params["id"] = id(self)
        return params
    def param_str(self, sep=", "):
        s = sep.join([k + "=" + "%2.2f" % v for k,v in self.params.items()])
        return s
    def set_runtime(self, time):
        self.runtime=time
    def set_result(self, name, res):
        self.results[name] = res
    def get_result(self, name):
        if name in self.results.keys():
            return self.results[name]
        return None
    def get_psi_cv(self):
        psi = self.results["PSI"]
        sd, mean = np.std(psi), np.mean(psi)
        return sd/mean
    def get_psi_mean(self):
        return np.mean(self.results["PSI"])
    def plot(self, ax = None, scale = 1):
        if ax == None:
            fig, ax = plt.subplots(1,3, figsize=(15*scale,5*scale))
        else:
            ax = ax.subplots(1,3, figsize=(15*scale,5*scale))

        tt = self.raster
        #st_res =self.get_result("stochastic")
        st_res_rasterized = self.get_result("stoch_rastr")
        
        ode_res = self.get_result("ODE")
        psi_res = self.get_result("PSI")
        lines = []
        lines.append(ax[0].plot(st_res_rasterized[:,0],st_res_rasterized[:,1], label = 'pre-RNA', color = color_pre, lw = line_width,drawstyle = 'steps')[0])
        lines.append(ax[0].plot(st_res_rasterized[:,0],st_res_rasterized[:,2], label = 'Incl', color = color_Incl,lw = line_width,drawstyle = 'steps')[0])
        lines.append(ax[0].plot(st_res_rasterized[:,0],st_res_rasterized[:,3], label = 'Skip', color = color_Skip,lw = line_width,drawstyle = 'steps')[0])
        #lines.append(ax[i,0].plot(st_res_rasterized[:,0],st_res_rasterized[:,1], label = 'Rasterized', color = "red",lw = line_width,drawstyle = 'steps')[0])
        ax[0].yaxis.set_label_coords(-0.28,0.25)
        ax[0].set_ylabel(self.param_str("\n"), rotation=0, fontsize="large" )
        ax[0].plot(tt,ode_res[:,0], color = color_pre, lw = 1*line_width, label = 'pre-RNA, ODE solution')
        ax[0].plot(tt,ode_res[:,1], color = color_Incl, lw = 1*line_width, label = 'Incl, ODE solution')
        ax[0].plot(tt,ode_res[:,2], color = color_Skip, lw = 1*line_width, label = 'Skip, ODE solution')
        ax[0].set_title(self.name)
        ax[1].hist(st_res_rasterized[:,(2,3)],label = ("Incl", "Skip"), orientation = "horizontal", color = (color_Incl, color_Skip))
        ax[1].set_title("Splice product distribution")
        bins_count = 15
        sd, mean = np.std(psi_res), np.mean(psi_res)
        n, bins, patches = ax[2].hist(psi_res, bins_count, orientation = "horizontal")
        #print(bins, mean, sd)
        y = st.norm.pdf(bins, mean, sd ) * (len(psi_res)*(bins[1]-bins[0]))
        ax[2].plot( y, bins, '--', lw=1, label='norm pdf')
        #print((mean, sd))
        ax[2].set_title("PSI distribution (CV = %2.2f)" % (sd/mean))
        

class Rates():
    def __init__(self, state=np.zeros(0)):
        self.state = state
        self.funcs = []
        self.reacts = np.zeros((0,len(self.state)))
    def set_state(self, state):
        self.state = state
    def add_rate_func(self,func,reaction=np.zeros(0)):
        self.funcs.append(func)
        reacts = self.reacts.tolist()
        reacts.append(reaction)
        self.reacts = np.array(reacts)
    def get_rates(self, state=None):
        rates =[]
        if state is None:
            state=self.state
        for f in self.funcs:
            rates.append(f(state))
        return np.array(rates)
    def get_reacts(self):
        return self.reacts

def solve_ODE_transcription_splicing(x,t,rates_obj):
   
    x_arr= x.tolist()
    x_arr.insert(0,t)
    x_arr = np.array(x_arr)
    #print(t)
    reacts = rates_obj.get_reacts()
    rates = rates_obj.get_rates(x_arr)
    
    dx=np.zeros(len(x))
    
    i=0
    for a in x:
        dx[i] = (rates*reacts[:,i+1]).sum()
        i+=1
    
    return dx
  
# analytical solution based on the equations above

def stoch_sim_transcription_splicing(rates_obj,initial_state,tf):
   
    # reaction matrix, the systems state includes the time points of the reactions in the first column
    reactions = rates_obj.get_reacts()
    # initialise the systems state
    
    state = np.hstack((0, initial_state))
    STATE = []
    STATE.append(state.copy())
    rates_obj.set_state(state)
    
    tt = 0
    while tt <= tf:
        # sample two random numbers uniformly between 0 and 1
        rr = sp.random.uniform(0,1,2)
        
        a_s = rates_obj.get_rates()
        #print(a_s)
        a_0 = a_s.sum()
        #print(a_0,a_s.sum())
            
        # time step: Meine Version
        tt = tt - sp.log(1. - rr[0])/a_0
        
        state[0] = tt
        
        # find the next reaction
        prop = rr[1] * a_0
        cum_a_s = np.cumsum(a_s)
        #print(cum_a_s, prop)
        ind = np.where(prop <= cum_a_s)[0][0]
        
        # update the systems state
        state +=reactions[ind]
        
        STATE.append(state.copy())
    
    return np.array(STATE)
# define parameters  

def sigmoid(x, mu=0, y_bounds=(0,1), range_95=6):
    y=1/(1+np.exp(6/range_95*(-x+mu)))
    y=y*(y_bounds[1]-y_bounds[0])+y_bounds[0]
    return y


def rasterize(x, steps):
    rows, cols = len(steps), x.shape[1]
    max_len = len(x)
    res = np.zeros((rows,cols))
    i=1
    j=0
    #print(steps)
    for step in steps:
        
        if step >= x[i-1,0] and step <= x[-1,0]:
            while  i < max_len and not(step >= x[i-1,0] and step < x[i,0]):
                i+=1
            if i==0:
                res[0] = x[0]
            else:
                #linear interpolation
#                factor = (step-x[i-1,0])/(x[i,0]-x[i-1,0])
#                res[j] = factor * (x[i]-x[i-1]) + x[i-1]
                res[j] = x[i-1]
        res[j,0] = step
        j+=1
            
    return res
        
def simulate(params, ODE = True):
    
    results={}
    mu=10
    sd = 2
    #prot_c = np.random.normal(mu,sd)
    #s1 = sigmoid(prot_c, mu,(0,2),range_95=2*sd)
    
    #s1,s2 = (1.2857718003999865, 0.7142281996000135)
    
    k_syn,s1,s2,s3,d0,d1,d2,d3= (params["constants"])
    #print((k_syn,s1,s2,s3,d0,d1,d2,d3))
    rates_obj = Rates()
   
#        rates_obj.add_rate_func(lambda st:  k_syn*(np.sin(st[0]/30*math.pi)+1.5), [0,1,0,0,0] )
    rates_obj.add_rate_func(lambda st:  k_syn, [0,1,0,0,0] )
    rates_obj.add_rate_func(lambda st:  d0*st[1], [0,-1,0,0,0] )
    rates_obj.add_rate_func(lambda st:  s1*st[1],  [0,-1,1,0,0] )
    rates_obj.add_rate_func(lambda st:  d1*st[2], [0,0,-1,0,0] )
    rates_obj.add_rate_func(lambda st:  s2*st[1] ,  [0,-1,0,1,0] )
    rates_obj.add_rate_func(lambda st:  d2*st[3], [0,0,0,-1,0] )
    rates_obj.add_rate_func(lambda st:  s3*st[1], [0,-1,0,0,1] )
    rates_obj.add_rate_func(lambda st:  d3*st[4],  [0,0,0,0,-1] )
    
    # calculate steady states
    #steady_state_RNA = k_1/d_1
    #steady_state_protein = k_2*k_1/d_1/d_2
    
    steady_state_pre = k_syn/(d0+s1+s2+s3)
    steady_state_Incl = steady_state_pre*s1/d1
    steady_state_Skip = steady_state_pre*s2/d2
    
    
    initial_state = params["init_state"]
    
    tt = params["raster"]
    # simulation time
    tf = params["runtime"]
    sim_st = stoch_sim_transcription_splicing(rates_obj,initial_state,tf)
    results["stochastic"] = sim_st
    sim_st_raster = rasterize(sim_st, tt)
    results["stoch_rastr"] = sim_st_raster
    
    #analyze the last third
    start_ind = int(len(sim_st_raster)*1/3)
    incl_counts = sim_st_raster[start_ind:,2]
    skip_counts = sim_st_raster[start_ind:,3]
    
    psi = incl_counts/(incl_counts+skip_counts)
    np.nan_to_num(psi, False)
    results["PSI"] = psi
    
    #determinisitic simulation
    
    y_0 = initial_state
    if ODE:
        sol_deterministic = odeint(solve_ODE_transcription_splicing,y_0,tt,args = (rates_obj,))
        results["ODE"] = sol_deterministic
    
    return results
        
def plot_simulations(sims, file_name = None):
    psi = []
    scale= 1
    fig, ax = plt.subplots(len(sims),3, figsize=(15*scale,5*len(sims)*scale))
    i=0
    for sim in sims:
        tt = sim.raster
        #st_res =sim.get_result("stochastic")
        st_res_rasterized = sim.get_result("stoch_rastr")
        
        ode_res = sim.get_result("ODE")
        psi_res = sim.get_result("PSI")
        lines = []
        lines.append(ax[i,0].plot(st_res_rasterized[:,0],st_res_rasterized[:,1], label = 'pre-RNA', color = color_pre, lw = line_width,drawstyle = 'steps')[0])
        lines.append(ax[i,0].plot(st_res_rasterized[:,0],st_res_rasterized[:,2], label = 'Incl', color = color_Incl,lw = line_width,drawstyle = 'steps')[0])
        lines.append(ax[i,0].plot(st_res_rasterized[:,0],st_res_rasterized[:,3], label = 'Skip', color = color_Skip,lw = line_width,drawstyle = 'steps')[0])
        #lines.append(ax[i,0].plot(st_res_rasterized[:,0],st_res_rasterized[:,1], label = 'Rasterized', color = "red",lw = line_width,drawstyle = 'steps')[0])
        ax[i,0].yaxis.set_label_coords(-0.28,0.25)
        ax[i,0].set_ylabel(sim.param_str("\n"), rotation=0, fontsize="large" )
        ax[i,0].plot(tt,ode_res[:,0], color = color_pre, lw = 1*line_width, label = 'pre-RNA, ODE solution')
        ax[i,0].plot(tt,ode_res[:,1], color = color_Incl, lw = 1*line_width, label = 'Incl, ODE solution')
        ax[i,0].plot(tt,ode_res[:,2], color = color_Skip, lw = 1*line_width, label = 'Skip, ODE solution')
        ax[i,0].set_title(sim.name)
        ax[i,1].hist(st_res_rasterized[:,(2,3)],label = ("Incl", "Skip"), orientation = "horizontal", color = (color_Incl, color_Skip))
        ax[i,1].set_title("Splice product distribution")
        bins_count = 15
        sd, mean = np.std(psi_res), np.mean(psi_res)
        n, bins, patches = ax[i,2].hist(psi_res, bins_count, orientation = "horizontal")
        #print(bins, mean, sd)
        y = st.norm.pdf(bins, mean, sd ) * (len(psi_res)*(bins[1]-bins[0]))
        ax[i,2].plot( y, bins, '--', lw=1, label='norm pdf')
        #print((mean, sd))
        ax[i,2].set_title("PSI distribution (CV = %2.2f)" % (sd/mean))
        
        psi.append(st_res_rasterized[:,1]/(st_res_rasterized[:,2]+st_res_rasterized[:,1]))
        
        i+=1
    
    fig.legend(lines, ("pre-RNA","Incl","Skip"),loc='lower left')
    
    if file_name != None:
        fig.savefig(file_name + ".png", format="png",dpi=200)
    

#unfinished(useless) function
def plot_psi_to_cv(sims):
    scale = 2
    fig, ax = plt.subplots(1, figsize=(5*scale,5*scale))
    colors = cm.rainbow(np.linspace(0, 1, len(sims)))
    for sims_row, col in zip(sims, colors) :
        cvs = [s.get_psi_cv() for s in sims_row]
        psis = [s.get_psi_mean() for s in sims_row]
#        psis = [s.expected_psi for s in sims_row]
        
        sds = [np.std(s.results["PSI"]) for s in sims_row]
        
        ax.plot(psis, cvs, label = "v_syn: %2.2f" % sims_row[0].params["k_syn"], color = col)
        ax.plot(psis, sds, "--", color = col)
        ax.set_ylabel("CV")
        ax.set_xlabel("PSI(mean)")
        ax.legend()

def plot_sd_over_t(sims, value = "PSI"):
    scale = 2
    fig, ax = plt.subplots(1, figsize=(5*scale,5*scale))
    colors = cm.rainbow(np.linspace(0, 1, len(sims)))
    for sim, col in zip(sims, colors) :
        psi = sim.results[value]
        indices = range(10,len(psi), 5)
        sds = [np.std(psi[0:l]) for l in indices]
#        ax.plot(indices, sds, label = "v_syn: %2.2f" % sim.params["k_syn"], color = col)
        ax.plot(indices, sds, label = sim.param_str(), color = col)
#        ax.plot(indices, psi[indices], "--", color=col, )
        
        ax.set_ylabel("SD (psi)")
        ax.set_xlabel("sim length")
        ax.legend()


def run_sims(sims, proc_count = 1):
    
    q1 = Queue()
    q2 = Queue()
    procs =[]
    for i in range(0,proc_count):
        print("creating process %d" % i )
        p = Process(target = child_process_func, args=(q1,q2))
        procs.append(p)
        p.start()
        print("Process %d started" % i)
    
    ids_to_sim = {}
    for sim in sims:
        params = sim.get_all_params()
        print("Putting sim %s in the queue" % sim.name)
        q1.put(params)
        ids_to_sim[id(sim)] = sim
    
    # send end signal to child processes
    for p in procs:
        q1.put(False)
    
    #get results
    for i in sims:
        res = q2.get()
        sim = ids_to_sim[res["id"]] 
        sim.results = res
        print("Results acquired for %s\n %s" % (sim.name, sim.param_str()))
    
    [p.join() for p in procs]
    
    

def child_process_func(q1,q2):
    
    while True:
        params = q1.get()
        if params==False:
            break
        res=simulate(params)
        res["id"] = params["id"]
        q2.put(res)

if __name__ == '__main__':
    
    start = time.time()
    sims = []
    flattened_sims = []
    shape = (9,9)
    
    #create simulations
    for i in range(0,shape[0]):
        sims.append([])
        k_syn = 0 + 1.5**i
        
        for j in range(0, shape[1]):
            
#            psi = (j+1)/(shape[1] + 1)
#            incl_ss = 5
#            skip_ss = incl_ss/psi -incl_ss
#            ret_ss = 1
#            s1 = 1
#            s2 = 1
#            s3 = 0.1
#            d0 = 0.1
#    
#            pre_ss = k_syn/(s1+s2+s3+d0)
#            
#            d1=s1*pre_ss/incl_ss
#            d2=s2*pre_ss/skip_ss
#            d3 = s3*pre_ss/ret_ss
#            
            s1 = 0 + 1.4**j
            s2= 8
            s3 = 1
            d1 = d2 = d3 = 1
            d0= 0.1
            
            pre_ss = k_syn/(s1 + s2 + s3 + d0)
            incl_ss = pre_ss/d1*s1
            skip_ss = pre_ss/d2*s2
            
            psi = incl_ss/(incl_ss+skip_ss) 
            
            #psi = incl_ss/(incl_ss+skip_ss)
            
            name="k_syn=%2.2f, s1=%2.2f (psi:%1.2f, I:%2.2f, S:%2.2f)" % (k_syn, s1, psi, incl_ss, skip_ss)
            s = SimParam(name,1000,
                         {"k_syn": k_syn, "s1": s1, "s2": s2, "s3": s3, "d0": d0, "d1": d1, "d2": d2, "d3": d3},
                         {"pre_RNA": 0, "Incl": 0, "Skip": 0, "ret": 0})
            sims[i].append(s)
            flattened_sims.append(s)
            s.expected_psi = psi
            
            print("Created sim: %s" % s.param_str())
        
    
    run_sims(flattened_sims,4)
        
    psi=plot_psi_to_cv(sims)
    end = time.time()
    print("runtime: %f s" % (end - start))

if False:
    PIK = "sims_titration_s1_v_syn.dat"
    
    with open(PIK, "wb") as f:
        pickle.dump(sims, f)
    with open(PIK, "rb") as f:
        sims= pickle.load(f)
        
    
    
def plot_3d(sims, value="PSI", axes = ("k_syn", "s1")):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    x = np.zeros((len(sims), len(sims[0])))
    y = np.copy(x)
    z = np.copy(x)
    
    i = 0
    for ss in sims:
        j=0
        for s in ss:
            x[i,j] = s.params[axes[0]]
            y[i,j] = s.params[axes[1]]
            if value == "CV":
                z[i,j] = s.get_psi_cv()
            else:
                z[i,j] = s.get_psi_mean()
            j+=1
        i+=1
    
    #ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    ax.plot_surface(x,y,z,  cmap=cm.coolwarm)
    #ax.plot_wireframe(x,y,z,  cmap=cm.coolwarm)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_zlabel(value)
    
    plt.show()


#fig,ax = plt.subplots(2,2, figsize = (14,15))
#ax[0,0].plot(st_res[:,0],st_res[:,1], label = 'pre-RNA', color = color_pre, lw = line_width,drawstyle = 'steps')
#ax[0,0].plot(tt,sol_deterministic[:,0], color = color_pre, lw = 1*line_width, label = 'pre-RNA, ODE solution')
#ax[0,0].plot(tt,sol_deterministic[:,1], color = color_Incl, lw = 1*line_width, label = 'Incl, ODE solution')
#ax[0,0].plot(tt,sol_deterministic[:,2], color = color_Skip, lw = 1*line_width, label = 'Skip, ODE solution')
#ax[0,0].plot(st_res[:,0],st_res[:,2], label = 'Incl', color = color_Incl,lw = line_width,drawstyle = 'steps')
#ax[0,0].plot(st_res[:,0],st_res[:,3], label = 'Skip', color = color_Skip,lw = line_width,drawstyle = 'steps')
##ax[0].plot(tt,sol_deterministic[:,1], color = color_Incl, lw = 2*line_width, label = 'Protein, ODE solution')
#ax[0,0].plot(st_res[:,0],np.zeros(len(st_res[:,0])) + steady_state_pre,'--', label = 'steady state pre-RNA',
#          color = color_pre)
#ax[0,0].plot(st_res[:,0],np.zeros(len(st_res[:,0])) + steady_state_Incl,'--', label = 'steady state Incl',
#          color = color_Incl)
#ax[0,0].plot(st_res[:,0],np.zeros(len(st_res[:,0])) + steady_state_Skip,'--', label = 'steady state Skip',
#          color = color_Skip)
#ax[0,0].legend(loc = 'best')
#ax[0,0].set_xlabel('Time')
#ax[0,0].set_ylabel('Molecule number')
##ax[0,0].set_ylim(0,2*steady_state_pre)
#ax[0,0].set_xlim(0,210)
#
#ax[0,1].plot(st_res[:,1],st_res[:,2], label = 'state trajectory', lw = line_width,color = color_phase_space)
#ax[0,1].plot(steady_state_pre,steady_state_Incl,'o',ms = 15,label = 'steady state', color = color_steady_state)
#ax[0,1].plot(pre_0,Incl_0,'o',ms = 15,label = 'Initial state', color = color_initial_state)
#ax[0,1].legend(loc = 'best')
#ax[0,1].set_ylabel('Incl')
#ax[0,1].set_xlabel('pre')
#ax[0,1].set_title('Phase space')

#ax[1,1].hist(psi_mean_last_third, label="PSI")
#ax[1,1].set_xlabel("psi=Incl/(Incl+Skip)")
#
#ax[1,0].hist(st_res[:,(1,2,3)], label=("pre","Incl","Skip"))

#heatmap
#psi = np.array(psi)
#names = (s.name for s in sims)
#
#fig, ax = plt.subplots()
#ax.set_yticklabels(names)
#im = ax.imshow(psi, aspect="auto", )
#cbar = ax.figure.colorbar(im, ax=ax)
#cbar.ax.set_ylabel("PSI", rotation=-90, va="bottom")
#
#fig.savefig("heat_map" + ".png", format="png",dpi=250)

if False:
    s = SimParam(name,100,
                         {"k_syn": 10, "s1": 10, "s2": 10, "s3": 10, "d0": 1, "d1": 1, "d2": 1, "d3": 1},
                         {"pre_RNA": 0, "Incl": 0, "Skip": 0, "ret": 0})
    
    simulate(s.get_all_params())
