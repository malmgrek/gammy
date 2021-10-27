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

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Installation](#installation)
- [Examples](#examples)
    - [Polynomial regression on 'roids](#polynomial-regression-on-roids)
        - [Predicting with model](#predicting-with-model)
        - [Plotting results](#plotting-results)
        - [Saving model on hard disk](#saving-model-on-hard-disk)
    - [Gaussian process regression](#gaussian-process-regression)
        - [More covariance kernels](#more-covariance-kernels)
        - [Defining custom kernels](#defining-custom-kernels)
    - [Non-linear manifold regression](#non-linear-manifold-regression)
    - [B-Spline basis](#b-spline-basis)
- [Testing](#testing)
- [Package documentation](#package-documentation)
- [ToDo](#todo)

<!-- markdown-toc end -->

## Installation

The package is found in PyPi.

``` shell
pip install gammy
```

## Examples

In this overview, we demonstrate the package's most important features through
common usage examples. 

### Polynomial regression on 'roids

A typical simple (but sometimes non-trivial) modeling task is to estimate an
unknown function from noisy data. First we import the bare minimum dependencies to be used in the below examples:

```python
>>> from functools import reduce

>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import pandas as pd

>>> import gammy
>>> from gammy.models.bayespy import GAM

```

Let's simulate a dataset:

```python
>>> np.random.seed(42)

>>> n = 30
>>> input_data = 10 * np.random.rand(n)
>>> y = 5 * input_data + 2.0 * input_data ** 2 + 7 + 10 * np.random.randn(n)

```

The object `x` is just a convenience tool for defining input data maps
as if they were just Numpy arrays:

```python
>>> from gammy.arraymapper import x

```

Define and fit the model:

```python
>>> a = gammy.formulae.Scalar(prior=(0, 1e-6))
>>> b = gamme.formulae.Scalar(prior=(0, 1e-6))
>>> bias = gammy.formulae.Scalar(prior=(0, 1e-6))
>>> formula = a * x + b * x ** 2 + bias
>>> model = GAM(formula).fit(input_data, y)

```

The model attribute `model.theta` characterizes the Gaussian posterior
distribution of the model parameters vector.

#### Predicting with model

```python
>>> np.round(model.predict(input_data[:3]), 4)
array([ 52.5711, 226.9461, 144.7863])

```

Predictions with uncertainty can be calculated as follows:

```python
>>> np.round(model.predict_variance(input_data[:3]), 4)
array([[ 52.5711, 226.9461, 144.7863],
       [ 79.3583,  95.1636,  79.9003]])

```

#### Plotting results

```python
>>> fig = gammy.plot.validation_plot(
...     model,
...     input_data,
...     y,
...     grid_limits=[0, 10],
...     input_maps=[x, x, x],
...     titles=["a", "b", "bias"]
... )

```

The grey band in the top figure is two times the prediction standard deviation
and, in the partial residual plots, two times the respective marginal posterior
standard deviation.

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example0-0.png)

It is also possible to plot the estimated Γ-distribution of the noise precision
(inverse variance) as well as the 1-D Normal distributions of each individual
model parameter.

Plot (prior or posterior) probability density functions of all model parameters:

```python
>>> fig = gammy.plot.gaussian1d_density_plot(model)

```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example0-1.png)

#### Saving model on hard disk

Saving:

<!-- NOTE: To skip doctests, one > has been removed -->
```python
>> model.save("/home/foobar/test.hdf5")
```

Loading:

<!-- NOTE: To skip doctests, one > has been removed -->
```python
>> model = GAM(formula).load("/home/foobar/test.hdf5")
```

### Gaussian process regression

Create fake dataset:

```python
>>> n = 50
>>> input_data = np.vstack((2 * np.pi * np.random.rand(n), np.random.rand(n))).T
>>> y = (
...     np.abs(np.cos(input_data[:, 0])) * input_data[:, 1] +
...     1 + 0.1 * np.random.randn(n)
... )

```

Define model:

``` python
>>> a = ExpSineSquared1d(
...     np.arange(0, 2 * np.pi, 0.1),
...     corrlen=1.0,
...     sigma=1.0,
...     period=2 * np.pi,
...     energy=0.99
... )
>>> bias = gammy.Scalar(prior=(0, 1e-6))
>>> formula = a(x[:, 0]) * x[:, 1] + bias
>>> model = gammy.models.bayespy.GAM(formula).fit(input_data, y)

```

Plot results:

``` python
>>> fig = gammy.plot.validation_plot(
...     model,
...     input_data,
...     y,
...     grid_limits=[[0, 2 * np.pi], [0, 1]],
...     input_maps=[x[:, 0:2], x[:, 1]],
...     titles=["a", "intercept"]
... )

```

Plot parameter probability density functions

``` python
>>> fig = gammy.plot.gaussian1d_density_plot(model)

```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example1-0.png)

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example1-1.png)

#### More covariance kernels

Target function: staircase shape with 5 steps between 0...1.

``` python
>>> input_data = np.arange(0, 1, 0.01)
>>> y = reduce(lambda u, v: u + v, [
...     1.0 * (input_data > c) for c in [0, 0.2, 0.4, 0.6, 0.8]
... ])

>>> grid = np.arange(0, 1, 0.01)
>>> corrlen = 0.01
>>> sigma = 2

>>> exp_squared_model = GAM(
...     gammy.formulae.ExpSquared1d(
...         grid=grid,
...         corrlen=corrlen,
...         sigma=sigma,
...         energy=0.9
...     )(x)
... ).fit(input_data, y)

>>> rat_quad_model = GAM(
...     gammy.formulae.RationalQuadratic1d(
...         grid=grid,
...         corrlen=corrlen,
...         alpha=1,
...         sigma=sigma,
...         energy=0.9
...     )(x)
... ).fit(input_data, y)

>>> orn_uhl_model = GAM(
...     gammy.formulae.OrnsteinUhlenbeck1d(
...         grid=grid,
...         corrlen=corrlen,
...         sigma=sigma,
...         energy=0.9
...     )(x)
... ).fit(input_data, y)

>>> np.round(exp_squared_model.mean_theta[0][:3], 2)
array([11.87,  5.09,  3.98])

>>> np.round(rat_quad_model.mean_theta[0][:3], 2)
array([9.54, 4.34, 2.48])

>>> np.round(orn_uhl_model.mean_theta[0][:3], 2)
array([12.86,  5.55,  4.3 ])

```

The plotting related boilerplate code is omitted:

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example1-2.png)

#### Defining custom kernels

It is straightforward to define custom formulas from "positive semidefinite" covariance kernel functions.

``` python
>>> def kernel(x1, x2):
...     """Kernel for min(x, x')
...
...     """
...     r = lambda t: t.repeat(*t.shape)
...     return np.minimum(r(x1), r(x2).T)


>>> grid = np.arange(0, 1, 0.001)
>>> Minimum = gammy.create_from_kernel1d(kernel)
>>> a = Minimum(grid=grid, energy=0.999)(x)

```

Let's compare to `ExpSquared1d`:

``` python
>>> b = ExpSquared1d(grid=grid, corrlen=0.05, sigma=1, energy=0.999)(x)


>>> def sample(X):
...     return np.dot(X, np.random.randn(X.shape[1]))


>>> _ = plt.plot(grid, sample(a.design_matrix(grid)), label="Custom")
>>> _ = plt.plot(grid, sample(b.design_matrix(grid)), label="Exp. squared")
>>> _ = plt.legend()

```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example1-3.png)

### Non-linear manifold regression

In this example we try estimating the bivariate "MATLAB function" using a
Gaussian process model with Kronecker tensor structure (see e.g.
[PyMC3](https://docs.pymc.io/notebooks/GP-Kron.html)). The main point in the
below example is that it is quite straightforward to build models that can learn
arbitrary 2D-surfaces.

Let us first create some artificial data using the MATLAB function!

```python
>>> n = 100
>>> input_data = np.vstack((
...     6 * np.random.rand(n) - 3, 6 * np.random.rand(n) - 3
... )).T
>>> y = (
...     gammy.utils.peaks(input_data[:, 0], input_data[:, 1]) + 
...     4 + 0.3 * np.random.randn(n)
... )

```

There is support for forming two-dimensional basis functions given two
one-dimensional formulas. The new combined basis is essentially the outer
product of the given bases. The underlying weight prior distribution priors and
covariances are constructed using the Kronecker product.

```python
>>> a = ExpSquared1d(
...     np.arange(-3, 3, 0.1),
...     corrlen=0.5,
...     sigma=4.0,
...     energy=0.99
... )(x[:, 0])  # NOTE: Input map is defined here!
>>> b = ExpSquared1d(
...     np.arange(-3, 3, 0.1),
...     corrlen=0.5,
...     sigma=4.0,
...     energy=0.99
... )(x[:, 1]) # NOTE: Input map is defined here!
>>> A = gammy.Kron(a, b)
>>> bias = Scalar(prior=(0, 1e-6))
>>> formula = A + bias
>>> model = GAM(formula).fit(input_data, y)

```

Note that same logic could be used to construct higher dimensional bases,
that is, one could define a 3D-formula:

<!-- NOTE: To skip doctests, one > has been removed -->
```python
>> formula_3d = gammy.Kron(gammy.Kron(a, b), c)

```

Finally, plot results:

```python
>>> fig = gammy.plot.validation_plot(
...     model,
...     input_data,
...     y,
...     grid_limits=[[-3, 3], [-3, 3]],
...     input_maps=[x, x[:, 0]],
...     titles=["A", "intercept"]
... )

```

Plot parameter probability density functions:

```
>>> fig = gammy.plot.gaussian1d_density_plot(model)

```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example2-0.png)

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example2-1.png)

The original function can be plotted as follows:

```python
>>> from mpl_toolkits.mplot3d import Axes3D

>>> X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
>>> Z = gammy.utils.peaks(X, Y) + 4

>>> fig = plt.figure()
>>> ax = fig.gca(projection="3d")
>>> _ = ax.plot_surface(X, Y, Z, color="r", antialiased=False)

```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/peaks.png)

### Spline regression

Constructing B-Spline based 1-D basis functions is also supported. Let's define
dummy data:

```python
>>> n = 30
>>> input_data = 10 * np.random.rand(n)
>>> y = 2.0 * input_data ** 2 + 7 + 10 * np.random.randn(n)

```

Define model:

``` python
>>> grid = np.arange(0, 11, 2.0)
>>> order = 2
>>> N = len(grid) + order - 2
>>> sigma = 10 ** 2
>>> formula = gammy.BSpline1d(
...     grid,
...     order=order,
...     prior=(np.zeros(N), np.identity(N) / sigma),
...     extrapolate=True
... )(x)
>>> model = gammy.models.bayespy.GAM(formula).fit(input_data, y)

```

Plot validation figure and parameter probability densities:

``` python
>>> fig = gammy.plot.validation_plot(
...     model,
...     input_data,
...     y,
...     grid_limits=[-2, 12],
...     input_maps=[x],
...     titles=["a"]
... )
>>> fig = gammy.plot.gaussian1d_density_plot(model)

```

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example3-0.png)

![](https://raw.githubusercontent.com/malmgrek/gammy/develop/doc/resources/example3-1.png)

## Testing

The package's unit tests can be ran with PyTest (`cd` to repository root):

``` shell
pytest -v
```

Running this documentation as a Doctest:

``` shell
python -m doctest -v README.md
```

## Package documentation

Documentation of the package with code examples:
<https://malmgrek.github.io/gammy>.

## TODO-list

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
