# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:35:50 2019

@author: timuhorn
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:18:55 2019

@author: timuhorn
"""

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from aux_th import *
import numpy as np
import os
import tempfile
import scipy.stats as st
import pyabc as pa


# Define a gaussian model

sigma = .5



area = 3
resolution = 100

x = np.linspace(-area , area , resolution)
y = np.linspace(-area , area , resolution)

X, Y = np.meshgrid(x, y)
Z = rastrigin([X, Y])

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 60, cmap='twilight_shifted')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z');


def model(parameters):
    # sample from a gaussian
    names = list(parameters.keys())
    x = np.array((parameters.x1, parameters.x2))
    y = rastrigin(x)
    # return the sample as dictionary
    return {"y": y}

# We define two models, but they are identical so far
models = [model, model]


# However, our models' priors are not the same.
# Their mean differs.
mu_x_1, mu_x_2 = 1, 2
parameter_priors = [
    pyabc.Distribution(x1=pyabc.RV("norm", mu_x_1, sigma), x2=pyabc.RV("norm", mu_x_1, sigma)),
    pyabc.Distribution(x1=pyabc.RV("norm", mu_x_1, sigma), x2=pyabc.RV("norm", mu_x_1, sigma))
]

# We plug all the ABC options together
abc = pyabc.ABCSMC(
    models, parameter_priors,
    pyabc.PercentileDistance(measures_to_use=["y"]))

# y_observed is the important piece here: our actual observation.
y_observed = 0
# and we define where to store the results
db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))
abc_id = abc.new(db_path, {"y": y_observed})

print("ABC-SMC run ID:", abc_id)

history = abc.run(minimum_epsilon=0.05, max_nr_populations=10)
model_probabilities = history.get_model_probabilities()
model_probabilities
pyabc.visualization.plot_model_probabilities(history)

#get extimated params 
df, w = history.get_distribution(m=0, t=1)

fig, ax = plt.subplots()
for t in range(history.max_t+1):
    df, w = history.get_distribution(m=0, t=t)
    pyabc.visualization.plot_kde_1d(
        df, w,
        xmin=0, xmax=5,
        x="x2", ax=ax,
        label="PDF t={}".format(t))
    ax_t = pyabc.visualization.plot_kde_2d(*history.get_distribution(m=0, t=t),
                     "x1", "x2",
                xmin=-1, xmax=4, numx=300,
                ymin=-1, ymax=4, numy=300)
#ax.axvline(observation, color="k", linestyle="dashed");
ax.legend();