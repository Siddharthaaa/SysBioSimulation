# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:35:50 2019

@author: timuhorn

Fitting counts-std(psi) dependence
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


def read_data():
    BRIE_dir = os.path.join("/home","timur","ext","working_dir","PRJEB15062", "BRIE_output")
    cell_paths = glob.glob(os.path.join(BRIE_dir, "*"))
    cell_names = [os.path.basename(c_n) for c_n in cell_paths]
    
    mux = pd.MultiIndex.from_product([cell_names, ["counts", "FPKM",  "PSI"]], names = ["cell_ID", "value"])
    
    summary_df = pd.DataFrame(columns=mux)
    
    cell_df = pd.DataFrame()
    for c_path in cell_paths:
        c_name = os.path.basename(c_path)
    #    print("read cell values for: "  + c_name)
        f_c_path = os.path.join(c_path, "fractions.tsv")
        cell_df_raw = pd.read_csv(f_c_path, "\t", low_memory = False)
        indexes = cell_df_raw.tran_id.str.contains("in$").values
        cell_df = cell_df_raw[indexes]
        cell_df_out = cell_df_raw[indexes == False]
        cell_df.counts = cell_df["counts"].values + cell_df_out.counts.values
        cell_df.FPKM = cell_df.FPKM.values + cell_df_out.FPKM.values
        
        summary_df[(c_name, "counts")] = cell_df["counts"]
        summary_df[(c_name, "FPKM")] = cell_df["FPKM"]
        summary_df[(c_name, "PSI")] = cell_df["Psi"]
        
    summary_df.set_index(cell_df["gene_id"].values, inplace=True)
#    df = perform_QC(summary_df, min_counts = 2e5, min_se = 3000, max_share=0.9,
#                    top_se_count=100, min_reads=5, min_cells=15)
    df_f_s = sth.perform_QC(summary_df, min_counts = 2e5, min_se = 2000, max_share=0.8,
                    top_se_count=100, min_reads=5, min_cells=40)
    sth.extend_data(df_f_s)
    return df_f_s


s = bs.get_exmpl_sim()
df_raw = read_data()
sth.extend_data(df_raw, True)
df = df_raw
#df = sth.filter_assumed_hkg(df_raw, psi = (0.1, 0.9), counts_max_cv = 0.8,
#                       min_counts_p = 0.2, min_psis_p = 0.2 )

counts = df["mean_counts"].values
psis = df["mean"].values
#psi_stds = df["std"].values

psis_all =  df.loc[:, (slice(None), "PSI")].values

psi_stds = np.nanstd(psis_all, axis = 1)
psis_all_stabilized = np.arcsin(np.sqrt(psis_all))
psi_stds_stabilized = np.nanstd(psis_all_stabilized, axis = 1)


m_i = 0
def model(parameters):
    global m_i
    m_i += 1
#    for k, v in parameters.items():
#        s.set_param(k,np.abs(v))
    seq_ql = parameters["seq_ql"]
    ex_pol = parameters["ex_pol"]
    print("Model eval: ", m_i)
    stds = sth.tmp_simulate_std_gillespie(counts, psis,
                                          sim_rnaseq=seq_ql,
                                          extrapolate_counts=ex_pol,
                                          var_stab = True)
    return {"y": stds}

models = [model]

class VectorDistance(pa.Distance):
    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        
        l_sum = 0
        for k in x.keys():
            l = np.linalg.norm(x[k] - x_0[k])
            l_sum += l
        return l_sum
            

# However, our models' priors are not the same.
# Their mean differs.

parameter_priors = [
    pa.Distribution(seq_ql=pa.RV("beta",3 ,3 ), ex_pol=pa.RV("beta", 3, 3))
]

# We plug all the ABC options together
abc = pa.ABCSMC(
    models, parameter_priors,
    VectorDistance(),
    population_size=100)
abc.max_number_particles_for_distance_update = 100

# y_observed is the important piece here: our actual observation.
y_observed = psi_stds
y_observed = psi_stds_stabilized
# and we define where to store the results

db_dir = tempfile.gettempdir()
db_dir = "./"
db_path = ("sqlite:///" +
           os.path.join(db_dir, "test.db"))
abc_id = abc.new(db_path, {"y": y_observed})

print("ABC-SMC run ID:", abc_id)

history = abc.run(minimum_epsilon=0.05, max_nr_populations=10)
model_probabilities = history.get_model_probabilities()
model_probabilities
pa.visualization.plot_model_probabilities(history)

#get extimated params 
#df, w = history.get_distribution(m=0, t=5)


fig, ax = plt.subplots()
for t in range(history.max_t+1):
    df_res, w = history.get_distribution(m=0, t=t)
    pa.visualization.plot_kde_1d(
        df_res, w,
        xmin=0, xmax=2,
        x="ex_pol", ax=ax,
        label="PDF t={}".format(t))
    ax_t = pa.visualization.plot_kde_2d(*history.get_distribution(m=0, t=t),
                     "seq_ql", "ex_pol",
                xmin=-0.1, xmax=1, numx=100,
                ymin=-0.1, ymax=1, numy=100)
ax_t = pa.visualization.plot_kde_2d(*history.get_distribution(m=0)
#ax.axvline(observation, color="k", linestyle="dashed");
ax.legend();

pa.visualization.plot_sample_numbers(history)
pa.visualization.plot_epsilons(history)

if False:
    res = sth.show_counts_to_variance(df, gillespie=False, log=False, keep_quantile=1,
                                    rnaseq_efficiency=0.115, extrapolate_counts=0.76,
                                    var_stab_std=True)
