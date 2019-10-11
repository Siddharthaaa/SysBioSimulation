# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:57:02 2019

@author: timuhorn
"""

"""
Paper title:
Combined single-cell profiling of expression and DNA methylation reveals splicing regulation and heterogeneity
"""

from bioch_sim import *
from aux_th import *
import support_th as sth
import pandas as pd
import numpy as np
import operator
import types
import re

import pyabc
#from gtfparse import read_gtf

s_file = "data/01_2_psi.xlsx"
tpm_file = "data/01_tpm_gene.csv"
counts_file = "data/01_counts_gene.csv"
gtf_file = "C:\\Users\\timuhorn\\Desktop\\Materials\\DBs\\gencode.v19.annotation.gtf_withproteinids"
map_file = "data/01_exon_to_gene.map"

#annot = read_gtf(gtf_file)

df1 = pd.read_excel(s_file, 0)
df2 = pd.read_excel(s_file, 1)
df3 = pd.read_excel(s_file, 2)

tpm_df = pd.read_csv(tpm_file, index_col=0 )
counts_df = pd.read_csv(counts_file, index_col=0 )


df_total = pd.merge(df1, df2, how="outer", on = "Exon_ID/Cell_ID")
phenotypes = ["iPS", "Endoderm", "Mouse", "iPS + Endodern"]
cols = df1.columns.values
cell_ids= cols[1:]

dfs = [df1, df2, df3, df_total]
phenotypes = ["iPS", "Endoderm", "Mouse", "iPS + Endodern"]
sumary = []

#for df in dfs: 
#    results = []
#    for i, row in df[1:].iterrows():
#        c = df.iloc[i,1:].values
#        c = np.array(c, dtype=np.float64)
#        all_val = c
#        c = c[np.logical_not(np.isnan(c))]
#        if(len(c)>10): # in more then 10 cells detected
#            obj = types.SimpleNamespace()
#            obj.score = get_multimodal_scores(c)[0][:,0].sum()
#            obj.values = c
#            obj.all_val = all_val
#            obj.exon_name = df.iloc[i,0]
#            obj.mean = np.mean(c)
#            obj.std = np.std(c)
#            results.append(obj)
#    sumary.append(results)

res_dfs =[]
tmp_count = 0

pattern = re.compile("^([^\\.]+).*")

for df, name  in zip(dfs, phenotypes):
    
    res_df = pd.DataFrame()
    #make theese fields able to contain arrays
    res_df['PSI_values'] = np.nan
    res_df['PSI_values'] = res_df['PSI_values'].astype(object)
    res_df['tpm'] = np.nan
    res_df['tpm'] = res_df['tpm'].astype(object)
    res_df['counts'] = np.nan
    res_df['counts'] = res_df['counts'].astype(object)
    
    
    cols = np.array(list(df))
    for i, row  in df.iterrows():
        c = df.iloc[i,1:].values
        c = np.array(c, dtype=np.float64)
        all_val = c
        c = c[np.logical_not(np.isnan(c))]
#        print(len(c))
        if(len(c)>50): # in more then 10 cells detected
            
            score = get_multimodal_scores(c)[0][:,0].sum()
            values = c
            exon_name = df.iloc[i,0]
            gene_name = df.iloc[i,0]
            mean = np.mean(c)
            std = np.std(c)
            np.zeros_like(3)
            res_df.loc[exon_name, "ID"] = exon_name
            res_df.loc[exon_name, "Bscore"] = score
            res_df.loc[exon_name, "mean"] = mean
            res_df.loc[exon_name, "std"] = std
            res_df.loc[exon_name, "gene"] = gene_name
            res_df.at[exon_name, "PSI_values"] = all_val
            exon_name_tmp = pattern.search(exon_name).group(1)
            
            #try to find tpm`s 
            if(exon_name_tmp in tpm_df.index):
                res_df.at[exon_name, "tpm"] = tpm_df.loc[exon_name_tmp, list(df)[1:]]
            else:
                res_df.at[exon_name, "tpm"] = None
                tmp_count+=1
            #try to find counts
            #counts are number of reads or transcripts
            if(exon_name_tmp in counts_df.index):
                
                res_df.at[exon_name, "counts"] = counts_df.loc[exon_name_tmp, list(df)[1:]]
            else:
                res_df.at[exon_name, "counts"] = None
                tmp_count+=1
    res_dfs.append(res_df)
df_original = res_dfs[0]
tmp = general_analysis(res_dfs[0])
tmp, i_p, df_corr = show_splicing_data(res_dfs[0], best_n = None, min_psi_th=0.20)

def convert_df(df, counts_df):
    cell_names = list(df)[1:]
    mux = pd.MultiIndex.from_product([cell_names, ["counts", "FPKM",  "PSI"]], names = ["cell_ID", "value"])
    summary_df = pd.DataFrame(columns=mux)
    
    indx = []
    for i, ind in enumerate(counts_df.index):
        if str(ind) in df1["Exon_ID/Cell_ID"].values:
            indx.append(i)
    
    indx = np.array(indx)
    for c_name in cell_names:
        summary_df[(c_name, "counts")] = counts_df[c_name].values
        summary_df[(c_name, "FPKM")] = counts_df[c_name].values
        print(len(summary_df))
        print(len(df))
        summary_df[(c_name, "PSI")] = df[c_name].values
    res_df = sth.extend_data(summary_df)
    return res_df

#fig, ax = plt.subplots(4, len(dfs))
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
#    names  = [res[i].exon_name for i in sort[-best_n:]]
#    
#    all_val = np.array([res[i].all_val for i in sort[-best_n:]])
#    df_bimodal_as = pd.DataFrame(all_val).transpose()
#    #heatmap
#    im = ax[3,i].imshow(df_bimodal_as.corr())
#    cbar = ax[3,i].figure.colorbar(im, ax=ax[3,i])
#    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
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
#
##plt.plot(res[sort[-20]].all_val, res[sort[-18]].all_val, ".")
#plt.plot(df_bimodal_as.iloc[:,0], df_bimodal_as.iloc[:,6], ".")

