import matplotlib.pyplot as plt
import numpy as np

import gammy
from gammy.arraymapper import x
from gammy.models.bayespy import GAM

# Data
input_data = np.linspace(0.01, 1, 50)
y = (
    1.3 + np.sin(1 / input_data) * np.exp(input_data) +
    0.1 * np.random.randn(50)
)

models = {
    # Polynomial model
    "polynomial": GAM(
        gammy.Polynomial(degrees=range(7))(x)
    ).fit(input_data, y),
    # Smooth Gaussian process model
    "squared_exponential": GAM(
        gammy.Scalar() * x +
        gammy.ExpSquared1d(
            grid=np.arange(0, 1, 0.05),
            corrlen=0.1,
            sigma=2
        )(x)
    ).fit(input_data, y),
    # Piecewise linear model
    "piecewise_linear": GAM(
        gammy.WhiteNoise1d(
            grid=np.arange(0, 1, 0.1),
            sigma=1
        )(x)
    ).fit(input_data, y)
}

# -----------------------------------------
# Posterior predictive confidence intervals
# -----------------------------------------
(fig, axs) = plt.subplots(1, 3, figsize=(8, 2))
for ((name, model), ax) in zip(models.items(), axs):
    # Posterior predictive mean and variance
    (μ, σ) = model.predict_variance(input_data)
    ax.scatter(input_data, y, color="r")
    ax.plot(input_data, model.predict(input_data), color="k")
    ax.fill_between(
        input_data,
        μ - 2 * np.sqrt(σ),
        μ + 2 * np.sqrt(σ),
        alpha=0.2
    )
    ax.set_title(name)

# ---------------------
# Posterior covariances
# ---------------------
(fig, axs) = plt.subplots(1, 3, figsize=(8, 2))
for ((name, model), ax) in zip(models.items(), axs):
    (ax, im) = gammy.plot.covariance_plot(
        model, ax=ax, cmap="rainbow"
    )
    ax.set_title(name)