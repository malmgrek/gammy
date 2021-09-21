Motivation
==========

Generalized additive models form a surprisingly general framework for building
models for both production software and scientific research. This Python package
offers tools for building the model terms as decompositions of various basis
functions. It is possible to model the terms e.g. as Gaussian processes (with
reduced dimensionality) of various kernels, as piecewise linear functions, and
as B-splines, among others. Of course, very simple terms like lines and
constants are also supported (these are just very simple basis functions).

The uncertainty in the weight parameter distributions is modeled using Bayesian
statistics with the help of the superb package
`BayesPy <http://www.bayespy.org/index.html>`_.
