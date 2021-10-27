import matplotlib.pyplot as plt
import numpy as np

import gammy
from gammy.arraymapper import x

# Define dummy data
n = 30
input_data = 10 * np.random.rand(n)
y = 2.0 * input_data ** 2 + 7 + 10 * np.random.randn(n)


# Define model
a = gammy.Scalar()

grid = np.arange(0, 11, 2.0)
order = 2
N = len(grid) + order - 2
sigma = 10 ** 2
a = gammy.BSpline1d(
    grid,
    order=order,
    prior=(np.zeros(N), np.identity(N) / sigma),
    extrapolate=True
)
formula = a(x)
model = gammy.models.bayespy.GAM(formula).fit(input_data, y)

#
# Plotting results --------------------------------------------
#

# Plot results
fig = gammy.plot.validation_plot(
    model,
    input_data,
    y,
    grid_limits=[-2, 12],
    input_maps=[x],
    titles=["a"]
)

# Plot parameter probability density functions
fig = gammy.plot.gaussian1d_density_plot(model)

plt.show()