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
    Cell Research
    Single-cell RNA-seq uncovers dynamic processes and critical regulators in mouse spermatogenesis
    https://www.nature.com/articles/s41422-018-0074-y
"""

from bioch_sim import *
from aux_th import *
import pandas as pd
import numpy as np
import scipy as sp
import operator
import types

splicing_file = "data\\04_GSE107644_INCLUSION_LEVELS_FULL.tab"



df_splicing = pd.read_csv(splicing_file, "\t")

dfs = []
phenotypes = ["TEST"]

df_tmp = df_splicing.filter(regex = "(GENE|EVENT|.*R1$)")

dfs.append(df_tmp)

res_df = pd.DataFrame()
res_df['PSI_values'] = np.nan
res_df['PSI_values'] = res_df['PSI_values'].astype(object)
sumary = []
for df, name  in zip(dfs, phenotypes):
    cols = np.array(list(df))
    for i, row  in df.iterrows():
        c = df.iloc[i,2:].values
        c = np.array(c, dtype=np.float64)
        all_val = c
        c = c[np.logical_not(np.isnan(c))]
#        print(len(c))
        if(len(c)>10): # in more then 10 cells detected
            
            score = get_multimodal_scores(c)[0][:,0].sum()
            values = c
            all_val = all_val
            exon_name = df.iloc[i,1]
            gene_name = df.iloc[i,0]
            mean = np.mean(c)
            std = np.std(c)
            
            res_df.loc[exon_name, "ID"] = exon_name
            res_df.loc[exon_name, "Bscore"] = score
            res_df.loc[exon_name, "mean"] = mean
            res_df.loc[exon_name, "std"] = std
            res_df.loc[exon_name, "gene"] = gene_name
            res_df.at[exon_name, "PSI_values"] = all_val
            
show_splicing_data(res_df)
#
#fig, ax = plt.subplots(3, len(dfs), squeeze = False)
#i = 0
#best_n = 20
#for df, res, name in zip(dfs, sumary, phenotypes):
#    
#    sort = np.argsort([res.score for res in res])
#    ax[0,i].set_title(name)
#    means = [obj.mean for obj in res] 
#    stds = [obj.std for obj in res]
#    scores = [obj.score for obj in res]
#    ax[0,i].plot(means, stds, ".", c = "blue")
#    plot_hist(res[sort[-40]].values, ax= ax[1,i])
#    ax[1,i].set_title("%s (Bs:%f)" % (res[sort[-1]].exon_name, res[sort[-1]].score))
#    ax[1,i].set_title("Best Bimodality: %s" % res[sort[-1]].exon_name)
#    ax[2,i].hist(scores)
#    ax[2,i].set_title("Bscore distr")
#    
#    means = [means[i] for i in sort[-best_n:]]
#    stds = [stds[i] for i in sort[-best_n:]]
#    names  = [res[i].gene_name for i in sort[-best_n:]]
#    
#    all_val = np.array([res[i].all_val for i in sort[-best_n:]])
#    df_bimodal_as = pd.DataFrame(all_val).transpose()
#    #heatmap
##    im = ax[3,i].imshow(df_bimodal_as.corr())
##    cbar = ax[3,i].figure.colorbar(im, ax=ax[3,i])
##    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
#    
#    ax[0,i].plot(means, stds, ".", c = "red")
#    
#    im, cbar = heatmap(df_bimodal_as.corr(),row_labels = names,
#                 col_labels =names , ax = None)
#    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
#    val_count = np.empty((best_n, best_n))
#    for k in range(best_n):
#        for j in range(best_n):
#            val_count[k,j] = (~np.isnan(all_val[k]) & ~np.isnan(all_val[j])).sum() 
#    annotate_heatmap(im, val_count, valfmt="{x:.0f}" )
#
#    i+=1

#plt.plot(res[sort[-20]].all_val, res[sort[-18]].all_val, ".")
#plt.plot(df_bimodal_as.iloc[:,10], df_bimodal_as.iloc[:,9], ".")
