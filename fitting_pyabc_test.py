# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:11:31 2019



@author: timuhorn
"""

from bioch_sim import *
import pyabc
import os

v_syn = 10
s1= 4
s2 = 1
s3 = 0.1

incl_ss = 10
skip_ss = 5
ret_ss = 3
d0 = 0.1
d1 = d2= 1
d3 = 1

pre_ss = v_syn/(s1+s2+s3+d0)
incl_ss=pre_ss*s1/d1
skip_ss=pre_ss*s2/d2
ret_ss=pre_ss*s3/d3

psi = incl_ss/(incl_ss + skip_ss)

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
s.expected_psi = psi

s.compile_system(True)

#start = time.time()
#for i in range(100):
#    s.simulate()
#    s.set_param("s1", 0.1)
#s.plot()
#duration = time.time()- start
#print("duration: ", duration)

#s.get_bimodality()

def model(parameters):
    for k, v in parameters.items():
        s.set_param(k,np.abs(v))
    s.simulate()
    y = s.get_psi_mean()
    y = s.get_bimodality(with_tendency=True)
    return {"y": y}


models = [model]

sigma = 1
sigma2 = 5
vsyn_expect = 20
s1_expect = 2
parameter_priors = [
    pyabc.Distribution(v_syn=pyabc.RV("norm", vsyn_expect, sigma2), s1=pyabc.RV("norm", s1_expect, sigma)),
]

# We plug all the ABC options together
abc = pyabc.ABCSMC(
    models, parameter_priors,
    pyabc.ZScoreDistance(measures_to_use=["y"]))
abc.max_number_particles_for_distance_update = 50

# y_observed is the important piece here: our actual observation.
val_expected = 0.6
# and we define where to store the results
db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))
abc_id = abc.new(db_path, {"y": val_expected})

print("ABC-SMC run ID:", abc_id)

populations = 10
history = abc.run(minimum_epsilon=0.1, max_nr_populations=populations)
model_probabilities = history.get_model_probabilities()

pyabc.visualization.plot_model_probabilities(history)

#get extimated params 
#df, w = history.get_distribution(m=0, t=5)

#fig1, ax_1d = plt.subplots()
fig =  plt.figure()
for t in range(history.max_t+1):
    ax = plt.subplot(2, 5, t+1)
    df, w = history.get_distribution(m=0, t=t)
#    pyabc.visualization.plot_kde_1d(
#        df, w,
#        xmin=0, xmax=10,
#        x="v_syn", ax=ax_1d,
#        label="PDF t={}".format(t))
    ax_t = pyabc.visualization.plot_kde_2d(*history.get_distribution(m=0, t=t),
                     "v_syn", "s1",
                xmin=1, xmax=25, numx=100,
                ymin=-3, ymax=3, numy=100,
                ax = ax)
#ax.axvline(observation, color="k", linestyle="dashed");
#ax.legend();
h = history.get_all_populations()
