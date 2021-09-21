import matplotlib.pyplot as plt
import numpy as np

import gammy
from gammy.arraymapper import x


def kernel(x1, x2):
    """Kernel for min(x, x')

    """
    r = lambda t: t.repeat(*t.shape)
    return np.minimum(r(x1), r(x2).T)

def sample(X):
    """Sampling from a GP kernel square-root matrix

    """
    return np.dot(X, np.random.randn(X.shape[1]))

grid = np.arange(0, 1, 0.001)

Minimum = gammy.create_from_kernel1d(kernel)
a = Minimum(grid=grid, energy=0.999)(x)

# Let's compare to exp squared
b = gammy.ExpSquared1d(
    grid=grid,
    corrlen=0.05,
    sigma=1,
    energy=0.999
)(x)

#
# Plotting results -----------------------------------------
#

ax = plt.figure().gca()
ax.plot(grid, sample(a.build_X(grid)), label="Custom")
ax.plot(grid, sample(b.build_X(grid)), label="Exp. squared")
ax.legend()

plt.show()