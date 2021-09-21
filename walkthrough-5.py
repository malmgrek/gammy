import matplotlib.pyplot as plt
import numpy as np

import gammy
from gammy.arraymapper import x


# Create data
n = 50
input_data = np.vstack((
    2 * np.pi * np.random.rand(n), np.random.rand(n)
)).T
y = (
    np.abs(np.cos(input_data[:, 0])) * input_data[:, 1]
    + 1 + 0.1 * np.random.randn(n)
)

# Define model
f = gammy.ExpSineSquared1d(
    np.arange(0, 2 * np.pi, 0.1),
    corrlen=1.0,
    sigma=1.0,
    period=2 * np.pi,
    energy=0.99
)
bias = gammy.Scalar()
formula = f(x[:, 0]) * x[:, 1] + bias
model = gammy.models.bayespy.GAM(formula).fit(input_data, y)

#
# Plot results ---------------------------------------------
#

# Plot results
fig1 = gammy.plot.validation_plot(
    model,
    input_data,
    y,
    grid_limits=[[0, 2 * np.pi], [0, 1]],
    input_maps=[x[:, 0:2], x[:, 1]],
    titles=["a", "intercept"]
)

# Plot parameter probability density functions
fig2 = gammy.plot.gaussian1d_density_plot(model)

plt.show()