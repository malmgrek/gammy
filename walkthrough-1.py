import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gammy
from gammy.arraymapper import x


# Define dummy data
n = 30
input_data = 10 * np.random.rand(n)
y = (
    5 * input_data + 2.0 * input_data ** 2 + 7 +
    10 * np.random.randn(n)
)

# Define model
a = gammy.Scalar()
b = gammy.Scalar()
bias = gammy.Scalar()
formula = a * x + b * x ** 2 + bias
model = gammy.models.bayespy.GAM(formula).fit(input_data, y)

#
# Plotting results -----------------------------------------
#

# Validation plot
fig1 = gammy.plot.validation_plot(
    model,
    input_data,
    y,
    grid_limits=[0, 10],
    input_maps=[x, x, x],
    titles=["a", "b", "bias"]
)

# Marginal posterior distributions of parameters
fig2 = gammy.plot.gaussian1d_density_plot(model)

plt.show()