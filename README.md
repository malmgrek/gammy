# Gammy – Generalized additive models in Python with a Bayesian twist

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/cover.png)

A Generalized additive model is a predictive mathematical model defined as a sum
of terms that are calibrated (fitted) with observation data. 

Generalized additive models form a surprisingly general framework for building
models for both production software and scientific research. This Python package
offers tools for building the model terms as decompositions of various basis
functions. It is possible to model the terms e.g. as Gaussian processes (with
reduced dimensionality) of various kernels, as piecewise linear functions, and
as B-splines, among others. Of course, very simple terms like lines and
constants are also supported (these are just very simple basis functions).

The uncertainty in the weight parameter distributions is modeled using Bayesian
statistical analysis with the help of the superb package
[BayesPy](http://www.bayespy.org/index.html). Alternatively, it is possible to
fit models using just NumPy.


### Other projects with GAM functionalities

- [PyGAM](https://pygam.readthedocs.io/en/latest/)
- [Statsmodels](https://www.statsmodels.org/dev/gam.html)

<!-- Remark for Emacs users: Table of contents comes out best, when generated at
the top of file -->

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Summary](#summary)
    - [Other projects with GAM functionalities](#other-projects-with-gam-functionalities)
- [Installation](#installation)
- [Examples](#examples)
    - [Polynomial regression on 'roids](#polynomial-regression-on-roids)
        - [Predicting with model](#predicting-with-model)
        - [Plotting results](#plotting-results)
        - [Saving model on hard disk for later use (HDF5)](#saving-model-on-hard-disk-for-later-use-hdf5)
    - [Gaussian process regression ("kriging")](#gaussian-process-regression-kriging)
        - [More kernel functions for GPs](#more-kernel-functions-for-gps)
        - [Define custom kernels](#define-custom-kernels)
    - [Multivariate Gaussian process regression](#multivariate-gaussian-process-regression)
    - [B-Spline basis](#b-spline-basis)
- [To-be-added features](#to-be-added-features)

<!-- markdown-toc end -->


## Installation

The package is found in PyPi.

``` shell
pip install gammy
```

## Features through examples

In this overview, we demonstrate the package's most important features through
common usage examples. 

### Polynomial regression on 'roids

Start with very simple dataset

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gammy

# NOTE: Used repetitively in defining model terms!
from gammy.arraymapper import x


np.random.seed(42)


# Define dummy data
n = 30
input_data = 10 * np.random.rand(n)
y = 5 * input_data + 2.0 * input_data ** 2 + 7 + 10 * np.random.randn(n)
```

The object `x` is just a convenience tool for defining input data maps
as if they were just Numpy arrays.

```python
# Define model
a = gammy.Scalar(prior=(0, 1e-6))
b = gammy.Scalar(prior=(0, 1e-6))
bias = gammy.Scalar(prior=(0, 1e-6))
formula = a * x + b * x ** 2 + bias
model = gammy.models.bayespy.GAM(formula).fit(input_data, y)
```

The model attribute `model.theta` characterizes the Gaussian posterior distribution of the model parameters vector.

#### Predicting with model

```python
model.predict(input_data[:3])
# array([  99.25493083,   23.31063443,  226.70702106])
```

Predictions with uncertainty can be calculated as follows (`scale=2.0` roughly corresponds to the 95% confidence interval):

```python
model.predict_total_uncertainty(input_data[:3], scale=2.0)
# (array([ 97.3527439 ,  77.79515549,  59.88285762]),
#  array([ 2.18915289,  2.19725385,  2.18571614]))
```

#### Plotting results

```python
# Plot results
fig = gammy.plot.validation_plot(
    model,
    input_data,
    y,
    grid_limits=[0, 10],
    input_maps=[x, x, x],
    titles=["a", "b", "bias"]
)
```

The grey band in the top figure is two times
the prediction standard deviation and, in the partial residual plots, two times
the respective marginal posterior standard deviation.

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example0-0.png)

It is also possible to plot the estimated Γ-distribution of the noise precision
(inverse variance) as well as the 1-D Normal distributions of each individual
model parameter.

```python
# Plot parameter probability density functions
fig = gammy.plot.gaussian1d_density_plot(model)
```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example0-1.png)

#### Saving model on hard disk for later use (HDF5)

Saving

```python
model.save("/home/foobar/test.hdf5")
```

Loading

```python
model = bayespy.GAM(formula).load("/home/foobar/test.hdf5")
```

### Gaussian process regression ("kriging")

```python
# Create some data
n = 50
input_data = np.vstack((2 * np.pi * np.random.rand(n), np.random.rand(n))).T
y = (
    np.abs(np.cos(input_data[:, 0])) * input_data[:, 1] 
    + 1 + 0.1 * np.random.randn(n)
)


# Define model
a = gammy.ExpSineSquared1d(
    np.arange(0, 2 * np.pi, 0.1),
    corrlen=1.0,
    sigma=1.0,
    period=2 * np.pi,
    energy=0.99
)
bias = gammy.Scalar(prior=(0, 1e-6))
formula = a(x[:, 0]) * x[:, 1] + bias
model = gammy.models.bayespy.GAM(formula).fit(input_data, y)


# Plot results
fig = gammy.plot.validation_plot(
    model,
    input_data,
    y,
    grid_limits=[[0, 2 * np.pi], [0, 1]],
    input_maps=[x[:, 0:2], x[:, 1]],
    titles=["a", "intercept"]
)


# Plot parameter probability density functions
fig = gammy.plot.gaussian1d_density_plot(model)
```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example1-0.png)

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example1-1.png)

#### More kernel functions for GPs

``` python

input_data = np.arange(0, 1, 0.01)

# Staircase function with 5 steps from 0...1
y = reduce(lambda u, v: u + v, [
    1.0 * (input_data > c) for c in [0, 0.2, 0.4, 0.6, 0.8]
])

grid = np.arange(0, 1, 0.001)
corrlen = 0.01
sigma = 2
a = gammy.ExpSquared1d(
    grid=grid,
    corrlen=corrlen,
    sigma=sigma,
    energy=0.999
)(x)
b = gammy.RationalQuadratic1d(
    grid=grid,
    corrlen=corrlen,
    alpha=1,
    sigma=sigma,
    energy=0.99
)(x)
c = gammy.OrnsteinUhlenbeck1d(
    grid=grid,
    corrlen=corrlen,
    sigma=sigma,
    energy=0.99
)(x)

exponential_squared = gammy.models.bayespy.GAM(a).fit(input_data, y)
rational_quadratic = gammy.models.bayespy.GAM(b).fit(input_data, y)
ornstein_uhlenbeck = gammy.models.bayespy.GAM(c).fit(input_data, y)
# Plot boilerplate ...

```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example1-2.png)

#### Define custom kernels

It is straightforward to define custom formulas from "positive semidefinite" covariance kernel functions.

``` python

def kernel(x1, x2):
    """Kernel for min(x, x')
    
    """
    r = lambda t: t.repeat(*t.shape)
    return np.minimum(r(x1), r(x2).T)


grid = np.arange(0, 1, 0.001)

Minimum = gammy.create_from_kernel1d(kernel)
a = Minimum(grid=grid, energy=0.999)(x)

# Let's compare to exp squared
b = gammy.ExpSquared1d(grid=grid, corrlen=0.05, sigma=1, energy=0.999)(x)


def sample(X):
    return np.dot(X, np.random.randn(X.shape[1]))


ax = plt.figure().gca()
ax.plot(grid, sample(a.build_X(grid)), label="Custom")
ax.plot(grid, sample(b.build_X(grid)), label="Exp. squared")
ax.legend()

```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example1-3.png)

### Multivariate Gaussian process regression

In this example we construct a basis corresponding to a multi-variate
Gaussian process with a Kronecker structure (see e.g. [PyMC3](https://docs.pymc.io/notebooks/GP-Kron.html)).

Let us first create some artificial data using the MATLAB function!

```python
# Create some data
n = 100
input_data = np.vstack((6 * np.random.rand(n) - 3, 6 * np.random.rand(n) - 3)).T


def peaks(x, y):
    """The MATLAB function

    """
    return (
        3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) -
        10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) -
        1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
    )


y = peaks(input_data[:, 0], input_data[:, 1]) + 4 + 0.3 * np.random.randn(n)
```

There is support for forming two-dimensional basis functions given two
one-dimensional formulas. The new combined basis is essentially the outer
product of the given bases. The underlying weight prior distribution priors and
covariances are constructed using the Kronecker product.

```python
# Define model
a = gammy.ExpSquared1d(
    np.arange(-3, 3, 0.1),
    corrlen=0.5,
    sigma=4.0,
    energy=0.99
)(x[:, 0])  # note that we need to define the input map at this point!
b = gammy.ExpSquared1d(
    np.arange(-3, 3, 0.1),
    corrlen=0.5,
    sigma=4.0,
    energy=0.99
)(x[:, 1]) # note that we need to define the input map at this point!
A = gammy.Kron(a, b)
bias = gammy.Scalar(prior=(0, 1e-6))
formula = A + bias
model = gammy.models.bayespy.GAM(formula).fit(input_data, y)
```

Note that same logic could be used to construct higher dimensional bases,
that is, one could define

```python
# 3-D formula
formula = gammy.kron(gammy.kron(a, b), c)
```

Finally, plot results.

```python
# Plot results
fig = gammy.plot.validation_plot(
    model,
    input_data,
    y,
    grid_limits=[[-3, 3], [-3, 3]],
    input_maps=[x, x[:, 0]],
    titles=["A", "intercept"]
)


# Plot parameter probability density functions
fig = gammy.plot.gaussian1d_density_plot(model)
```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example2-0.png)

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example2-1.png)

The original function can be plotted like so

```python
from mpl_toolkits.mplot3d import Axes3D


X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = peaks(X, Y) + 4

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_surface(X, Y, Z, color="r", antialiased=False)
```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/peaks.png)

### B-Spline basis

Constructing B-Spline based 1-D basis functions is also supported.

```python
# Define dummy data
n = 30
input_data = 10 * np.random.rand(n)
y = 2.0 * input_data ** 2 + 7 + 10 * np.random.randn(n)


# Define model
a = gammy.Scalar(prior=(0, 1e-6))

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
```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example3-0.png)

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example3-1.png)

## Testing

The package's unit tests can be ran with PyTest (`cd` to repository root):

``` shell
pytest -v
```

Running the tests in this documentation

## Documentation

Documentation of the package with code examples:
<https://malmgrek.github.io/gammy>.

## ToDo

- **TODO** Quick model template functions (e.g. splines, GPs)
- **TODO** Shorter overview and examples in README. Other docs inside `docs`.
- **TODO** Support indicator models in plotting
- **TODO** Fixed ordering for GP related basis functions.
- **TODO** Hyperpriors for model parameters – Start from diagonal precisions.
           Instead of `(μ, Λ)` pairs, the arguments could be just
           BayesPy node.
- **TODO** Support non-linear GAM models, fitting with autograd.
- **TODO** Multi-dimensional observations.
- **TODO** Dynamically changing models.
