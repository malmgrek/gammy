from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

import gammy
from gammy.arraymapper import x


# Define data
input_data = np.arange(0, 1, 0.01)
y = reduce(lambda u, v: u + v, [
    # Staircase function with 5 steps from 0...1
    1.0 * (input_data > c) for c in [0, 0.2, 0.4, 0.6, 0.8]
])

# Kernel parameters
grid = np.arange(0, 1, 0.01)
corrlen = 0.01
sigma = 2

# Define and fit models with different kernels
exp_squared_model = gammy.models.bayespy.GAM(
    gammy.ExpSquared1d(
        grid=grid,
        corrlen=corrlen,
        sigma=sigma,
        energy=0.9
    )(x)
).fit(input_data, y)
rat_quad_model = gammy.models.bayespy.GAM(
    gammy.RationalQuadratic1d(
        grid=grid,
        corrlen=corrlen,
        alpha=1,
        sigma=sigma,
        energy=0.9
    )(x)
).fit(input_data, y)
orn_uhl_model = gammy.models.bayespy.GAM(
    gammy.OrnsteinUhlenbeck1d(
        grid=grid,
        corrlen=corrlen,
        sigma=sigma,
        energy=0.9
    )(x)
).fit(input_data, y)

#
# Plotting results -----------------------------------------
#

ax = plt.figure(figsize=(10, 4)).gca()
ax.plot(input_data, y, label="Actual")
ax.plot(
    input_data,
    exp_squared_model.predict(input_data),
    label="Exp. squared"
)
ax.plot(
    input_data,
    rat_quad_model.predict(input_data),
    label="Rat. quadratic"
)
ax.plot(
    input_data,
    orn_uhl_model.predict(input_data),
    label="Ohrnstein-Uhlenbeck"
)
ax.legend()