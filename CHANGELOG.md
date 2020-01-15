# Change Log

## 0.0.2

### Added
- Formulas for `BSpline1d` and `ReLU`

### Changed
- Rename `KeyFunction` to `ArrayMapper`
- Remove `attr` frozen class feature from `BayesianGAM` in order to be able
  to define `theta` as the prior distribution by default in the initialization

## 0.0.1

### Added
- Create package basic structure
- Add `BayesPyFormula` for configuring and manipulating basis-function
  based GAM models
- Add `BayesianGAM` for fitting predictive GAM models using Bayesian
  methods
- Add utilities for constructing Gaussian process covariances using
  suitable kernel functions
- Add `kron` constructing multi-dimensional Gaussian processes with
  Kronecker covariance structure
- Add various plotting tools for validating and inspecting model parameters
  and their posterior probability distributions
- Add three illustrated examples to README.md

## X.Y.Z
### Added
### Changed
### Fixed
### Removed
