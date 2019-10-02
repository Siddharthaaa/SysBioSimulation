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


cores = 10
populations = 10

v0 = 20
spl_r = 0.33

s = bs.get_exmpl_sim()
s = bs.get_exmpl_sim("CoTrSplicing_2")
s.compile_system()


#s.set_param("spl_rate",2)
s.set_param("u1_1_br", 0.25)
s.set_param("u1_2_br", 0.016)   
s.set_param("u2_1_br", 0.0183)
s.set_param("u2_2_br", 1.25)
s.set_param("spl_rate", 0.55)
s.set_param("d1", 2.2e-4)
s.set_param("d2", 2.2e-4)
s.set_param("elong_v", v0)
#s.plot_par_var_1d("elong_v", np.linspace(10,100,51), s.get_psi_mean, ignore_fraction=0.5)
#s.simulate()
#s.plot_course(products=["Incl", "Skip"], products2=["ret", "ret_i1"], t_bounds=(100, 1234))

s.set_runtime(80000)
#
#psis = []
#for i in range(30):
#    s.simulate()
#    psis.append(s.get_psi_mean(ignore_fraction=0.5))
#
#print("PSI_mean: ", np.mean(psis))

#s.get_psi_mean(ignore_fraction = 0.5)


messured_points = 5
v0s = np.linspace(20,100,messured_points)
runtimes = np.linspace(80000, 20000, messured_points, dtype = int)


def model(parameters):
    v1_1 = parameters["u1_1_br"]
    v1_2 = parameters["u1_2_br"]
    v2_1 = parameters["u2_1_br"]
    v2_2 = parameters["u2_2_br"]    
    spl_r = parameters["spl_r"]
    
    s.set_param("u1_1_br", v1_1)
    s.set_param("u1_2_br", v1_2)
    s.set_param("u2_1_br", v2_1)
    s.set_param("u2_2_br", v2_2)
    s.set_param("spl_rate", spl_r)
#    s.set_param("elong_v", v0)
    psis =[]
    counts =[]
    for v0, r_time in zip(v0s, runtimes):
        s.set_param("elong_v", v0)
        s.set_runtime(r_time)
        s.simulate()
        psi = s.get_psi_mean(ignore_fraction = 0.7)
        if np.isnan(psi):
            psi= 0
        psis.append(psi)
        counts.append(np.mean(s.get_res_by_expr("Incl+Skip")[-3000:]))
#        ret_t.append(np.mean(s.get_res_from_expr("ret + ret_i1 + ret_i2")[-3000:]))
    return {"PSI": np.array(psis),
            "counts": np.array(counts)}

models = [model]

class y_Distance(pa.Distance):
    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        counts = x["counts"]
        counts_0 = x_0["counts"]
        counts_d = sth.norm_dist(counts, counts_0, 5, 2)
        psi_d = abs((x_0["PSI"] - x["PSI"]))
        
        res = np.linalg.norm(counts_d*0.5 + psi_d*3)
        return res
            

# However, our models' priors are not the same.
# Their mean differs.


limits = dict(u1_1_br = (0.1, 0.4),
              u1_2_br=(0.001, 0.04),
              u2_1_br=(0.001, 0.05),
              u2_2_br = (0.7, 2),
              spl_r=(0.2, 1.2)) 

parameter_priors = pa.Distribution(**{key: pa.RV("beta", 2, 2, a, b - a)
                                    for key, (a,b) in limits.items()})

# We plug all the ABC options together
abc = pa.ABCSMC(
    models, parameter_priors,
    y_Distance(),
    population_size=100
    ,    sampler=pa.sampler.MulticoreEvalParallelSampler(cores)
    )
abc.max_number_particles_for_distance_update = 100

# y_observed is the important piece here: our actual observation.
# search for psi == 0.5
y_observed = {"counts": np.linspace(20,100, messured_points),
              "PSI": np.linspace(1,0, messured_points)}
# and we define where to store the results

db_dir = tempfile.gettempdir()
db_dir = "./"
db_path = ("sqlite:///" +
           os.path.join(db_dir, "test.db"))
abc_id = abc.new(db_path, y_observed)

print("ABC-SMC run ID:", abc_id)

h = abc.run(minimum_epsilon=0.01, max_nr_populations=populations)
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
from scipy.stats import gaussian_kde
df, w = h.get_distribution(m=0)
grid = pa.visualization.plot_kde_matrix(df, w, limits=limits)

pa.visualization.plot_sample_numbers(h)
pa.visualization.plot_epsilons(h)
