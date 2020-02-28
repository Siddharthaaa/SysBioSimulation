#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:59:32 2019

@author: timur
"""

class SimParam(SimParam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_psi_cv(self, **kwargs):
        psi = self.compute_psi(**kwargs)[1]
        sd, mean = np.std(psi), np.mean(psi)
        return sd/mean
    def get_psi_mean(self, **kwargs):
        self.compute_psi(**kwargs)
        return np.mean(self.results["PSI"][1])
    def compute_psi(self, products = ["Incl", "Skip"], solution="stoch_rastr",
                    ignore_extremes = False, ignore_fraction=0.1, recognize_threshold = 1,
                    exact_sum = None, sim_rnaseq = None):
        
#        print("compute psi...")
        sim_st_raster = self.results[solution]
        start_ind = int(len(sim_st_raster)*ignore_fraction)
        incl_counts = np.array(self.get_res_col(products[0]), dtype=np.int32)
        skip_counts = np.array(self.get_res_col(products[1]), dtype=np.int32)
        
        if(sim_rnaseq is not None):
            incl_counts = sp.stats.binom.rvs(incl_counts, sim_rnaseq)
            skip_counts = sp.stats.binom.rvs(skip_counts, sim_rnaseq)
        
        indices =  np.array(np.where(incl_counts + skip_counts >= recognize_threshold))
        if ignore_extremes:
            indices_extr = [np.where((incl_counts != 0) * (skip_counts != 0))]
            indices = np.intersect1d(indices, indices_extr)
        if exact_sum is not None:
            indices_extr = [np.where(incl_counts + skip_counts == exact_sum)]
            indices = np.intersect1d(indices, indices_extr)
            
        indices = indices[np.where(indices >= start_ind)]
#        print(len(incl_counts), len(skip_counts))
        
        incl_counts = incl_counts[indices]
        skip_counts = skip_counts[indices]
        
        psi = incl_counts/(incl_counts+skip_counts)
#        psi = psi[np.logical_not(np.isnan(psi))]
#        np.nan_to_num(psi, False)
        #result contains times and values arrays
        self.results["PSI"] = np.array((sim_st_raster[indices,0], psi))
        return indices, psi
    
    def get_psi_end(self, products = ["Incl", "Skip"], res_type="stoch"):
        p = products
        incl = self.get_res_col(p[0], res_type)[-1]
        skip = self.get_res_col(p[1], res_type)[-1]
        psi = incl/(incl+skip)
        return psi
    
    def get_bimodality(self, name = "PSI", ignore_extremes=False, ignore_fraction=0.1, recognize_threshold=1, with_tendency=False):
        
        settings = (name, ignore_extremes, ignore_fraction, recognize_threshold, with_tendency)
        
        if not hasattr(self, 'bimodality'):
            self.bimodality = {}
        
        if settings in self.bimodality:
            return self.bimodality[settings]
        print("compute bimodality....")
        max_range = None
        if name == "PSI":
            self.compute_psi(ignore_extremes=ignore_extremes, ignore_fraction=ignore_fraction,
                             recognize_threshold=recognize_threshold)
            inp = self.results["PSI"][1]
            max_range = 1
        else:
            inp = self.get_res_col(name)
        res = get_multimodal_scores(inp, max_range=max_range)
        self.bimodality[settings] = (res[0])[:,0].max()
        if(with_tendency and name == "PSI"):
            return self.bimodality[settings] + np.std(self.results["PSI"][1])
        return self.bimodality[settings]