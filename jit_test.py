# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:03:39 2019

@author: imb30
"""

from bioch_sim import *
import numba as nb

spec = [
    ('number', nb.int64),
]

@nb.jitclass(spec)
class some_class:
    def __init__(self, something):
        self.number = something
    def get_num(self):
        return self.number

my_object = some_class(5)
print(my_object.get_num())

start = time.time()
s2 = SimParam("Circadian oscillation", 60000,60000,
                      {"k1":1,"k3":1,"k5":1,"k2":0.1,"k4":0.2,"k6":0.2,"K_i":1, "n":10},
                      
                      {"RNA":0, "protein":1, "protein_p":4})
s2.add_reaction("k1*K_i**n/(K_i**n+protein_p**n)",{"RNA":1})
s2.add_reaction("k2*RNA",{"RNA":-1})
s2.add_reaction("k3*RNA",{"protein":1})
s2.add_reaction("k4*protein",{"protein":-1})
s2.add_reaction("k5*protein",{"protein_p":1})
s2.add_reaction("k6*protein_p",{"protein_p":-1})
print(s2.compile_system())
#s2.simulate()
#s2.plot()

r_time1 = []
ttt=[]
for i in range(20):
    start = time.time()
    s2.runtime = i*1000
    s2.set_raster_count(i*1000)
    s2.simulate()
    r_time1.append(time.time() - start)
    ttt.append(s2.runtime) 
    
fig, ax = plt.subplots(1,1, figsize=(7,7))

ax.plot(ttt[:20], r_time2[:20], color = "green", label="with JIT", lw=2)
#ax.plot(ttt[:20], r_time3[:20], color = "red", label="without JIT", lw=2)
ax.legend()
ax.set_title("Performance mit JIT-Compiler")
ax.set_xlabel("Simulated time")
ax.set_ylabel("Runtime [s]")


print("Runtime %f" % (time.time() - start))
print(r_time1)

def testf(func):
    print("AAAAAAAAAAAAAAAAAAAA")
    func(np.zeros(0,0,0))
    
#compute_stochastic_evolution(np.array([[1,2,3],[2,3,4]]),
#                             np.array([1,2,3]),
#                             np.int(100), 
#                             np.float(1000), 
#                             np.float(1000))
    
@nb.njit
def jit_test(f, b):
    return f(b)
    
@nb.njit
def arg_func(b):
    a = np.cumsum(np.array([1,2,3,4]))
    print(a)
    return b**10