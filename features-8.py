import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import gammy
from gammy.arraymapper import x


# Create some data
n = 100
input_data = np.vstack((
    6 * np.random.rand(n) - 3, 6 * np.random.rand(n) - 3
)).T

def peaks(x, y):
    """The MATLAB function

    """
    return (
        3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) -
        10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) -
        1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
    )

y = (
    peaks(input_data[:, 0], input_data[:, 1]) + 4 +
    0.3 * np.random.randn(n)
)

# Plot the MATLAB function
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = peaks(X, Y) + 4
ax = plt.figure().add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, color="r", antialiased=False)
ax.set_title("Exact MATLAB function")

# Define model
a = gammy.ExpSquared1d(
    np.arange(-3, 3, 0.1),
    corrlen=0.5,
    sigma=4.0,
    energy=0.9
)(x[:, 0])
b = gammy.ExpSquared1d(
    np.arange(-3, 3, 0.1),
    corrlen=0.5,
    sigma=4.0,
    energy=0.9
)(x[:, 1])
A = gammy.Kron(a, b)
bias = gammy.Scalar()
formula = A + bias
model = gammy.models.bayespy.GAM(formula).fit(input_data, y)

#
# Plot results
#

# Validation plot
fig = gammy.plot.validation_plot(
    model,
    input_data,
    y,
    grid_limits=[[-3, 3], [-3, 3]],
    input_maps=[x, x[:, 0]],
    titles=["Surface estimate", "intercept"]
)

# Probability density functions
fig = gammy.plot.gaussian1d_density_plot(model)