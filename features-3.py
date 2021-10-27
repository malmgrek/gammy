import matplotlib.pyplot as plt
import numpy as np

import gammy
from gammy.arraymapper import x
from gammy.models.bayespy import GAM

n = 100
input_data = np.vstack(
    [2 * np.random.rand(n) - 1, 2 * np.random.rand(n) - 1]
).T
y = (
    input_data[:, 0] ** 3 -
    3 * input_data[:, 0] * input_data[:, 1] ** 2
)

# The model form can be relaxed with "black box" terms such as
# piecewise linear basis functions:
model = GAM(
    gammy.Polynomial(range(5))(x[:, 0]) +
    gammy.WhiteNoise1d(
        np.arange(-1, 1, 0.05),
        sigma=1
    )(x[:, 1]) * x[:, 0]
).fit(input_data, y)

# Let's check if the model was able to fit correctly:
fig = plt.figure(figsize=(8, 2))
(X, Y) = np.meshgrid(
    np.linspace(-1, 1, 100),
    np.linspace(-1, 1, 100)
)
Z = X ** 3 - 3 * X * Y ** 2
Z_est = model.predict(
    np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
).reshape(100, 100)
ax = fig.add_subplot(121, projection="3d")
ax.set_title("Exact")
ax.plot_surface(
    X, Y, Z, color="r", antialiased=False
)
ax = fig.add_subplot(122, projection="3d")
ax.set_title("Estimated")
ax.plot_surface(
    X, Y, Z_est, antialiased=False
)