#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:55:42 2020

@author: timur
"""


psis1 = [psi_analyticaly(vpol2, gene_length, l, m1, m2, k, n,ki,ks,kesc,kesc_r) for vpol2 in vpols]
psis2 = [psi_analyticaly(vpol2, gene_length, l, m1, m2, k, n,ki,ks,kesc,kesc_r) for vpol2 in vpols]
psis3 = [psi_analyticaly(vpol2, gene_length, l, m1, m2, k, n,ki,ks,kesc,kesc_r) for vpol2 in vpols]
psis4 = [psi_analyticaly(vpol2, gene_length, l, m1, m2, k, n,ki,ks,kesc,kesc_r) for vpol2 in vpols]
fig, axs = plt.subplots(2,2, figsize=(8,8))


axs[0,0].plot(vpols, psis1)
axs[1,1].set_xlabel("vpol")
axs[0,0].set_ylabel("PSI")
axs[1,0].set_xlabel("vpol")
axs[1,0].set_ylabel("PSI")

axs[0,1].plot(vpols, psis2)
axs[1,0].plot(vpols, psis3)
axs[1,1].plot(vpols, psis4)

axs[0,0].set_xscale("log")
axs[0,1].set_xscale("log")
axs[1,0].set_xscale("log")
axs[1,1].set_xscale("log")

fig.suptitle("Schematic representation of 4 types")


import numpy as np
import bioch_sim as bs
import matplotlib.pyplot  as plt

x = np.linspace(-60,100,100)
y1 = [bs.sin_proximity(x1,0,30,50) for x1 in x]
y2 = [bs.asym_proximity(x1,0,30,50,6) for x1 in x]

fig, ax = plt.subplots()

ax.plot(x,y1, lw=2, label = "Sinus effekt")
ax.plot(x,y2, lw=2, label = "Hill inhibition")
ax.axvline(0,ls = "--",c="black")
ax.set_ylabel("Inhibition")
ax.set_xlabel("Position")
ax.legend()

x = np.linspace(20,40,100)
y1 = [bs.asym_proximity(x1,30,1,10,6) for x1 in x]

fig, ax = plt.subplots()

ax.plot(x,y1, lw=2, label = "Hill func.")
ax.axvline(30,ls = "--",c="black")
ax.set_ylabel("Inhibition")
ax.set_xlabel("Position")
ax.legend()

