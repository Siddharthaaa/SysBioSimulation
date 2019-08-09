# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:20:49 2019

@author: timuhorn
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:57:02 2019

@author: timuhorn
"""

"""
Paper title:
    Single-cell transcriptomics reveals bimodality in expression and splicing in immune cells
"""

from bioch_sim import *
import pandas as pd
import numpy as np
import scipy as sp
import operator
import types

file = "data\\02_psi_sc_transcrip_table4.xls"

df_full = pd.read_excel(file, 0)
genes = df_full.iloc[1:,1].values
df = df_full.filter(regex="S\d+$")

results = []
for i, row in df[1:].iterrows():
    c = df.iloc[i,1:].values
    c = np.array(c, dtype=np.float64)
    c = c[np.logical_not(np.isnan(c))]
    if(len(c)>10): # in more then 10 cells detected
        obj = types.SimpleNamespace()
        obj.score = get_multimodal_scores(c)[0][:,0].sum()
        obj.values = c
        obj.gene_name = genes[i]
        obj.mean = np.mean(c)
        obj.std = np.std(c)
        results.append(obj)

res = results
fig, ax = plt.subplots(1, 3)
    
sort = np.argsort([res.score for res in res])
ax[0].set_title("BMDC")
means = [obj.mean for obj in res] 
stds = [obj.std for obj in res]
scores = [obj.score for obj in res]
ax[0].plot(means, stds, ".", c = "blue")
plot_hist(res[sort[-1]].values, ax= ax[1])
ax[1].set_title("%s (Bs:%f)" % (res[sort[-1]].gene_name, res[sort[-1]].score))
ax[1].set_title("Best Bimodality: %s" % res[sort[-1]].gene_name)
ax[2].hist(scores)
ax[2].set_title("Bscore distr")

means = [means[i] for i in sort[-5:]]
stds = [stds[i] for i in sort[-5:]]
ax[0].plot(means, stds, ".", c = "red")
i+=1

best1 = results[sort[-1]].values
best2 = results[sort[-2]].values
