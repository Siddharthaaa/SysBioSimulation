# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:18:55 2019

@author: timuhorn
"""

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from aux_th import *
import numpy as np

area = 3
resolution = 100

x = np.linspace(-area , area , resolution)
y = np.linspace(-area , area , resolution)

X, Y = np.meshgrid(x, y)
Z = rastrigin([X, Y])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='twilight_shifted')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');