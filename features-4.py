import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gammy
from gammy.arraymapper import x


# Simulated dataset
n = 30
input_data = np.random.rand(n)
y = (
   input_data ** 2 * np.sin(2 * np.pi * input_data) + 1 +
   0.1 * np.random.randn(n)  # Simulated pseudo-random noise
)

# Define model
f = gammy.ExpSquared1d(
    grid=np.arange(0, 1, 0.05),
    corrlen=0.1,
    sigma=0.01,
    energy=0.99
)
c = gammy.Scalar()
formula = f(x) + c
model = gammy.models.bayespy.GAM(formula).fit(input_data, y)

#
# Plotting results -----------------------------------------
#

# Plot validation plot
fig1 = gammy.plot.validation_plot(
   model,
   input_data,
   y,
   grid_limits=[0, 1],
   input_maps=[x, x],
   titles=["f", "c"]
)

# Parameter posterior density plot
fig2 = gammy.plot.gaussian1d_density_plot(model)

plt.show()