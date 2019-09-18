#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:46:41 2019

@author: timur

fitting Cotranscriptional splicing
"""




from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import numpy as np
import os
import tempfile
import scipy.stats as st
import pyabc as pa
import pandas as pd
import support_th as sth
import bioch_sim as bs
import glob


v0 = 55
v1 = 0.6
v2 = 0.05
spl_r = 0.33

s = bs.get_exmpl_sim("CoTrSplicing")
s.set_param("spl_rate", 0.3)
#s.set_param("u1_1_br", v1)
s.set_param("u1_2_br", v2)   
s.set_param("u2_1_br", v2)
s.set_param("u2_2_br", v1)
#s.set_param("spl_rate", spl_r)
#s.set_param("elong_v", v0)
s.set_runtime(40000)

psis = []
#for i in range(100):
#    s.simulate()
#    psis.append(s.get_psi_mean(ignore_fraction=0.5))
#
#print(np.median(psis))

#s.plot_course(products=["Skip","Incl", "ret", "ret_i1"], res = ["stoch"])
#s.plot_course(products=["U2_Pol", "U2_2"], res = ["stoch"])
#s.get_psi_mean(ignore_fraction = 0.5)

def model(parameters):
    v1 = parameters["v1"]
    v2 = parameters["v2"]
#    v0 = parameters["v0"]
    spl_r = parameters["spl_r"]
#    s.set_param("u1_1_br", v1)
    s.set_param("u1_2_br", v2)
    s.set_param("u2_1_br", v2)
    s.set_param("u2_2_br", v1)
#    s.set_param("elong_v", v0)
    s.set_param("spl_rate", spl_r)
    s.simulate()
    res = s.get_psi_mean(ignore_fraction = 0.8)
    if np.isnan(res):
        res = 0
    return {"y": res}

models = [model]

class y_Distance(pa.Distance):
    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        res = np.abs(x["y"] - x_0["y"])
        return res
            

# However, our models' priors are not the same.
# Their mean differs.


limits = dict(v1=(0, 2),
              v2=(0, 0.15),
#              v0=(10, 100),
              spl_r=(0, 1))
bound1 = 0.001
bound2 = 10
parameter_priors = pa.Distribution(**{key: pa.RV("beta", 2, 2, a, b - a)
                                    for key, (a,b) in limits.items()})
#parameter_priors = [
#    pa.Distribution(v1=pa.RV("beta",3 ,3, bound1, bound2 ),
#                    v2=pa.RV("beta", 3, 3, bound1, bound2))]

# We plug all the ABC options together
abc = pa.ABCSMC(
    models, parameter_priors,
    y_Distance(),
    population_size=100,
    sampler=pa.sampler.MulticoreEvalParallelSampler(15))
abc.max_number_particles_for_distance_update = 100

# y_observed is the important piece here: our actual observation.
# search for psi == 0.5
y_observed = 0.5
# and we define where to store the results

db_dir = tempfile.gettempdir()
db_dir = "./"
db_path = ("sqlite:///" +
           os.path.join(db_dir, "test.db"))
abc_id = abc.new(db_path, {"y": y_observed})

print("ABC-SMC run ID:", abc_id)

h = abc.run(minimum_epsilon=0.01, max_nr_populations=10)
model_probabilities = h.get_model_probabilities()
pa.visualization.plot_model_probabilities(h)

#get extimated params 
#df, w = h.get_distribution(m=0, t=5)

#fig, ax = plt.subplots()
#for t in range(h.max_t+1):
#    df_res, w = h.get_distribution(m=0, t=t)
#    pa.visualization.plot_kde_1d(
#        df_res, w,
#        xmin=0, xmax=2,
#        x="v1", ax=ax,
#        label="PDF t={}".format(t))
#    ax_t = pa.visualization.plot_kde_2d(*h.get_distribution(m=0, t=t),
#                     "v1", "v2",
#                xmin=bound1, xmax=bound2, numx=100,
#                ymin=bound1, ymax=bound2, numy=100)
#ax.axvline(observation, color="k", linestyle="dashed");
#ax.legend();

df, w = h.get_distribution(m=0)
pa.visualization.plot_kde_matrix(df, w, limits=limits)

pa.visualization.plot_sample_numbers(h)
pa.visualization.plot_epsilons(h)
