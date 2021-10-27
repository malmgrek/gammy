import matplotlib.pyplot as plt
import numpy as np

import gammy
from gammy.arraymapper import x
from gammy.models.bayespy import GAM


# Simulate data
input_data = np.linspace(0.01, 1, 50)
y = (
   1.3 + np.sin(1 / input_data) * np.exp(input_data) +
   0.1 * np.random.randn(50)
)

# A totally non-sense model, just an example
bias = gammy.Scalar()
slope = gammy.Scalar()
k = gammy.Scalar()
formula = bias + slope * x + k * x ** (1/2)
model_bad = GAM(formula).fit(input_data, y)

# Ideal model with a user-defined function basis
basis = [
    lambda t: np.sin(1 / t) * np.exp(t),
    lambda t: np.ones(len(t))
]

formula = gammy.Formula(
    terms=[basis],
    # mean and inverse covariance (precision matrix)
    prior=(np.zeros(2), 1e-6 * np.eye(2))
)
model_ideal = GAM(formula).fit(input_data, y)

plt.scatter(input_data, y, c="r", label="data")
plt.plot(input_data, model_bad.predict(input_data), label="bad")
plt.plot(input_data, model_ideal.predict(input_data), label="ideal")
plt.legend()

plt.show()