# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:27:10 2019

@author: Timur
"""
from numba import cuda
import numba as nb
import numpy as np
pointer = None

def test_loc_func():
    global pointer
    a = 5
    gpu_pow = cuda.jit(lambda x:  x*x, device=True)
    
    @cuda.jit("void(float64[:,:], float64[:,:])")
    def _loc_func(b, arr):
        nonlocal a 
        aa = 3
        arr1 = arr[0]
        b[1,1]=gpu_pow(a*aa + len(arr1))
        arr = cuda.local.array((10,10), dtype = nb.float64)
        arr[0,4] += 4
        return None
    
    pointer= _loc_func
    
    print(a)

a_t = np.zeros((10,10), dtype=float)
test_loc_func()
d_a_t = cuda.to_device(a_t)
pointer[1,10](d_a_t, np.zeros((10,10), dtype = np.float64) +3)
d_a_t.to_host()
print(a_t)
