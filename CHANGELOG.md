# Change Log

## 0.0.4

### Removed
- Dependence from Pandas

## 0.0.3

### Fixed
- Simplify `BayesPyFormula.__mul__` so that all bases are just multiplied
  with the given input map
- All references to `sigma` now correspond to variance

## 0.0.2

### Added
- Formulas for `BSpline1d` and `ReLU`

### Changed
- `KeyFunction` is now called `ArrayMapper`

### Remove
- `attr` frozen class feature from `BayesianGAM` in order to be able
  to define `theta` as the prior distribution by default in the initialization

## 0.0.1

### Added
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
### Added
### Changed
### Fixed
### Removed
