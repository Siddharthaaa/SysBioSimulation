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


v0 = 100
spl_r = 0.031

s = bs.get_exmpl_sim("CoTrSplicing")
#s.set_param("spl_rate",2)
s.set_param("u1_1_br", 0.03)
s.set_param("u1_2_br", 0.16)   
s.set_param("u2_1_br", 0.086)
s.set_param("u2_2_br", 0.152)
s.set_param("spl_rate", 0.136)
s.set_param("d1", 2.2e-4)
s.set_param("d2", 2.2e-4)
s.set_param("elong_v", v0)



s.set_runtime(80000)
#
#psis = []
#for i in range(100):
#    s.simulate()
#    psis.append(s.get_psi_mean(ignore_fraction=0.5))
#
#print(np.mean(psis))
#
#s.plot_course(products=["Skip","Incl", "ret", "ret_i1"], res = ["stoch"])
#s.plot_course(products=["U2_Pol", "U2_2"], res = ["stoch"])
#s.get_psi_mean(ignore_fraction = 0.5)


messured_points = 5
v0s = np.linspace(20,100,messured_points)
runtimes = np.linspace(100000, 20000, messured_points, dtype = int)


def model(parameters):
    v1 = parameters["u1_2_br"]
    v2 = parameters["u2_1_br"]
    v3 = parameters["u2_2_br"]
    spl_r = parameters["spl_r"]
    d = parameters["d"]
#    s.set_param("u1_1_br", v1)
    s.set_param("u1_2_br", v1)
    s.set_param("u2_1_br", v2)
    s.set_param("u2_2_br", v3)
    s.set_param("d1", d)
    s.set_param("d2", d)
    s.set_param("spl_rate", spl_r)
#    s.set_param("elong_v", v0)
    psis =[]
    counts =[]
    ret_t = []
    for v0, r_time in zip(v0s, runtimes):
        s.set_param("elong_v", v0)
        s.set_runtime(r_time)
        s.simulate()
        psi = s.get_psi_mean(ignore_fraction = 0.7)
        if np.isnan(psi):
            psi= 0
        psis.append(psi)
        counts.append(np.mean(s.get_res_from_expr("Incl+Skip")[-3000:]))
        ret_t.append(np.mean(s.get_res_from_expr("ret + ret_i1 + ret_i2")[-3000:]))
    return {"PSI": np.array(psis),
            "counts": np.array(counts),
            "ret_total":np.array(ret_t)}

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
        ret = x["ret_total"]
        ret_0 = x_0["ret_total"]
        ret_d = sth.norm_dist(ret, ret_0, ret_0/5, 2) 
        psi_d = abs((x_0["PSI"] - x["PSI"]))
        
        res = np.linalg.norm(ret_d*0.5 + counts_d + psi_d*2)
        return res
            

# However, our models' priors are not the same.
# Their mean differs.


limits = dict(u1_2_br=(0.01, 0.3),
              u2_1_br=(0.001, 0.1),
              u2_2_br = (0.01, 0.2),
              spl_r=(0.01, 0.2),
              d = (1e-5,1e-3))
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
    sampler=pa.sampler.MulticoreEvalParallelSampler(10))
abc.max_number_particles_for_distance_update = 100

# y_observed is the important piece here: our actual observation.
# search for psi == 0.5
y_observed = {"counts": np.linspace(20,100, messured_points),
              "PSI": np.linspace(1,0, messured_points),
              "ret_total":np.linspace(4,20,messured_points)}
# and we define where to store the results

db_dir = tempfile.gettempdir()
db_dir = "./"
db_path = ("sqlite:///" +
           os.path.join(db_dir, "test.db"))
abc_id = abc.new(db_path, y_observed)

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

params = dict(pr_on=2.000000e+00,
pr_off=1.000000e-01,
elong_v=5.000000e+01,
gene_len=3.000000e+03,
spl_rate=7.100000e-02,
u1_1_bs_pos=1.500000e+02,
u1_2_bs_pos=1.700000e+03,
u1_1_br=3.000000e-02,
u1_1_ur=1.000000e-03,
u1_2_br=1.620000e-01,
u1_2_ur=1.000000e-03,
u2_1_bs_pos=1.500000e+03,
u2_2_bs_pos=2.800000e+03,
u2_1_br=3.900000e-02,
u2_2_br=1.020000e-01,
u2_1_ur=1.000000e-03,
u2_2_ur=1.000000e-03,
tr_term_rate=1.000000e+02,
d0=2.000000e-04,
d1=2.000000e-04,
d2=2.000000e-04,
d3=1.000000e-03)
