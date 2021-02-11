# Change Log


## 0.3.2

### Remove
- Dependence from `attrs` package


## 0.3.1

### Add
- Common constructor for kernel based GP formulas
- Unit tests for GP kernels

### Fix
- Numpy testing incompatibility issues
- Numpy deprecation warning


## 0.3.0

### Add
- Type hints for most functions
- Unit testing module for modules `formulae`, `utils`
- Unit tests for `BayesianGAM`
- Docstrings


## 0.2.3

### Add
- Parameter `verbose` to `BayesianGAM.fit` defaulting to `False`


## 0.2.2

### Add
- Support serializing models to JSON


## 0.2.1

### Add
- Methods `predict_marginal` and `predict_marginals` to
  `BayesianGAM`
- Type hints to `BayesianGAM`


## 0.2.0

### Change
- `BayesPyFormula` => `Formula`
- Formulae package => module

### Fix
- `__div__` to `__truediv__` in `ArrayMapper`

### Remove
- Broken comparison methods in `ArrayMapper`

### Add
- Unit tests for `ArrayMapper`


## 0.1.0

### Change
- `BayesPyFormula` interface: Attribute `priors` => `prior`.
  This gives more flexibility in defining priors. E.g., when summing up terms,
  sometimes one wants the final prior not to be block diagonal.
- Naming `*partial*` => `*marginal*` in `BayesianGAM`

### Add
- `Sum` and `Flatten` transformations for `BayesPyFormula`
- Smoke tests corresponding the README examples


## 0.0.4

### Remove
- Dependence from Pandas


## 0.0.3

### Fix
- Simplify `BayesPyFormula.__mul__` so that all bases are just multiplied
  with the given input map
- All references to `sigma` now correspond to variance


## 0.0.2

### Add
- Formulas for `BSpline1d` and `ReLU`

### Change
- `KeyFunction` is now called `ArrayMapper`

### Remove
- `attr` frozen class feature from `BayesianGAM` in order to be able
  to define `theta` as the prior distribution by default in the initialization


## 0.0.1

### Add
- Create package basic structure
- `BayesPyFormula` for configuring and manipulating basis-function
  based GAM models
- `BayesianGAM` for fitting predictive GAM models using Bayesian
  methods
- Utilities for constructing Gaussian process covariances using
  suitable kernel functions
- `kron` constructing multi-dimensional Gaussian processes with
  Kronecker covariance structure
- Various plotting tools for validating and inspecting model parameters
  and their posterior probability distributions
- Three illustrated examples to README.md

## X.Y.Z
### Add
### Change
### Fix
### Remove
