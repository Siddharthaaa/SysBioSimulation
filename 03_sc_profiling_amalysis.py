# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:41:53 2019

@author: timuhorn
"""

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

splicing_file = "data\\03_GSE85908_splicing.csv"
meta_file = "data\\03_GSE85908_metadata.csv"
exon_file = "data\\03_GSE85908_splicing_feature.csv"


df_splicing = pd.read_csv(splicing_file, ",")
df_meta = pd.read_csv(meta_file, ",")
df_exon = pd.read_csv(exon_file, ",")
df_exon.set_index("event_id", inplace = True)


df_full = pd.merge(df_splicing, df_meta, how = "left", on = "sample_id")

#filter 
df_filtered = df_full[df_full.outlier ==False]
df_filtered = df_filtered[df_filtered.single == True]

#get phenotypes
phenotypes = df_full["phenotype"].unique()

dfs = []
for pheno in phenotypes:
    df_tmp = df_filtered[df_filtered.phenotype == pheno]
    df_tmp = df_tmp.filter(regex = ".*exon.*")
    dfs.append(df_tmp)

dfs.append(df_filtered.filter(regex = ".*exon.*"))
phenotypes = np.append(phenotypes, "ALL")


sumary = []
for df, name  in zip(dfs, phenotypes):
    cols = list(df)
    results = []
    for i in cols:
        c = df[i].values
        c = np.array(c, dtype=np.float64)
        c = c[np.logical_not(np.isnan(c))]
        if(len(c)>10): # in more then 10 cells detected
            obj = types.SimpleNamespace()
            obj.score = get_multimodal_scores(c)[0][:,0].sum()
            obj.values = c
            obj.exon_name = i
            obj.gene_name = df_exon.loc[i,"ensembl_id"]
            obj.mean = np.mean(c)
            obj.std = np.std(c)
            results.append(obj)
    sumary.append(results)

fig, ax = plt.subplots(3, len(dfs))
i = 0
for df, res, name in zip(dfs, sumary, phenotypes):
    
    sort = np.argsort([res.score for res in res])
    ax[0,i].set_title(name)
    means = [obj.mean for obj in res] 
    stds = [obj.std for obj in res]
    scores = [obj.score for obj in res]
    ax[0,i].plot(means, stds, ".", c = "blue")
    plot_hist(res[sort[-1]].values, ax= ax[1,i])
    ax[1,i].set_title("%s (Bs:%f)" % (res[sort[-1]].gene_name, res[sort[-1]].score))
    ax[1,i].set_title("Best Bimodality: %s" % res[sort[-1]].gene_name)
    ax[2,i].hist(scores)
    ax[2,i].set_title("Bscore distr")
    
    means = [means[i] for i in sort[-20:]]
    stds = [stds[i] for i in sort[-20:]]
    ax[0,i].plot(means, stds, ".", c = "red")
    i+=1
