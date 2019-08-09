# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:40:22 2019

@author: timuhorn
"""

file_name = "data\\01_sceset.run_20287.salmon.preqc_gene.rds"

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

readRDS = robjects.r['readRDS']
df = readRDS(file_name)
pandas2ri.activate()
df = pandas2ri.ri2py(df) #does not work: Conversion 'ri2py' not defined for objects of type '<class 'rpy2.robjects.methods.RS4'>'
