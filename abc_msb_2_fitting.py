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


s = bs.get_exmpl_sim()
df = read_data()

counts = df["mean_counts"].values
psis = df["mean"].values
psi_stds = df["std"].values
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
                                          extrapolate_counts=ex_pol)
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
    VectorDistance())

# y_observed is the important piece here: our actual observation.
y_observed = psi_stds
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
df, w = history.get_distribution(m=0, t=5)

fig, ax = plt.subplots()
for t in range(history.max_t+1):
    df, w = history.get_distribution(m=0, t=t)
    pa.visualization.plot_kde_1d(
        df, w,
        xmin=0, xmax=2,
        x="ex_pol", ax=ax,
        label="PDF t={}".format(t))
    ax_t = pa.visualization.plot_kde_2d(*history.get_distribution(m=0, t=t),
                     "seq_ql", "ex_pol",
                xmin=-0.1, xmax=1, numx=100,
                ymin=-0.1, ymax=1, numy=100)
#ax.axvline(observation, color="k", linestyle="dashed");
ax.legend();

def read_data():
    excl_path = "dates/msbfig3.xlsx"
    
    
    counts_df = pd.read_excel(excl_path, 1)   
    psi_df = pd.read_excel(excl_path, 2)
    cell_types = counts_df["cell.population"].values 
    cell_types = np.unique(cell_types)
        
    res_dfs = []
    
    cell_ids = counts_df.filter(regex="^\d+$")
    
    for cell_t in cell_types:
    #        res_df = pd.DataFrame()
        mux = pd.MultiIndex.from_product([cell_ids, ["counts", "FPKM",  "PSI"]], names = ["cell_ID", "value"])
        res_df = pd.DataFrame(columns=mux)
        for df , v_name in zip ([psi_df, counts_df, counts_df], ["PSI", "counts", "FPKM"]):
            
            filtered_df = df[counts_df["cell.population"] == cell_t]
            
            genes = filtered_df["gene.name"].values
            
            c_df = filtered_df.filter(regex="^\d+$")
            
            for col in c_df:
                res_df[(col, v_name)] = c_df[col].values if v_name != "counts" else np.exp(c_df[col].values)
    #                res_df[(col, v_name)] = c_df[col].values 
                res_df.set_index(genes, inplace=True)
    #        perform_QC(res_df, min_counts=100,min_se= 10, max_share=1, top_se_count=1, min_reads=5, min_cells=10 )
        res_dfs.append(res_df)
    
    d = res_dfs[1]
    d = sth.perform_QC(d, min_counts=100,min_se= 10, max_share=1, top_se_count=1, min_reads=5, min_cells=10 )
    sth.extend_data(d)
    return d
