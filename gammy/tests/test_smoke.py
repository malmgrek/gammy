"""Sanity checks for semi-realistic modeling cases

"""

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

import gammy
from gammy.arraymapper import x


def test_polynomial(fit_model):
    np.random.seed(42)
    input_data = 10 * np.random.rand(30)
    y = (
        5 * input_data +
        2.0 * input_data ** 2 +
        7 +
        10 * np.random.randn(len(input_data))
    )
    formula = (
        gammy.Scalar(prior=(0, 1e-6)) * x +
        gammy.Scalar(prior=(0, 1e-6)) * x ** 2 +
        gammy.Scalar(prior=(0, 1e-6))
    )
    model = fit_model(formula, gammy.numpy.Delta(0.01342))(input_data, y)
    assert_almost_equal(
        model.predict(input_data[[0, 3, 6, 9, 12]]),
        np.array([52.5711, 104.2863, 14.4829, 136.9858, 180.0926]),
        decimal=3
    )
    return


def test_gp(fit_model):
    np.random.seed(42)
    n = 50
    input_data = np.vstack(
        (
            2 * np.pi * np.random.rand(n),
            np.random.rand(n),
        )
    ).T
    y = (
        np.abs(np.cos(input_data[:, 0])) * input_data[:, 1] +
        1 +
        0.1 * np.random.randn(n)
    )
    formula = gammy.ExpSineSquared1d(
        np.arange(0, 2 * np.pi, 0.1),
        corrlen=1.0,
        sigma=1.0,
        period=2 * np.pi,
        energy=0.99
    )(x[:, 0]) * x[:, 1] + gammy.Scalar(prior=(0, 1e-6))
    model = fit_model(formula, gammy.numpy.Delta(100))(input_data, y)
    assert_almost_equal(
        model.predict(input_data[[1, 42, 11, 26, 31]]),
        np.array([1.7891, 1.8710, 1.3014, 1.2922, 1.3159]),
        decimal=3
    )
    return


def test_kron(fit_model):
    np.random.seed(42)
    n = 30
    input_data = np.vstack(
        (
            6 * np.random.rand(n) - 3,
            6 * np.random.rand(n) - 3,
        )
    ).T

    def peaks(x, y):
        """The function in Mathworks logo

        """
        return (
            3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) -
            10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) -
            1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
        )

    y = peaks(
        input_data[:, 0],
        input_data[:, 1]
    ) + 4 + 0.3 * np.random.randn(n)
    formula = gammy.Kron(
        gammy.ExpSquared1d(
            np.arange(-3, 3, 0.1),
            corrlen=0.5,
            sigma=4.0,
            energy=0.99
        )(x[:, 0]),
        gammy.ExpSquared1d(
            np.arange(-3, 3, 0.1),
            corrlen=0.5,
            sigma=4.0,
            energy=0.99
        )(x[:, 1])
    ) + gammy.Scalar(prior=(0, 1e-6))
    model = fit_model(formula, gammy.numpy.Delta(24.2289))(input_data, y)
    assert_almost_equal(
        model.predict(input_data[[1, 5, 12, 19], :]),
        np.array([3.7859, 3.5640, 3.7391, 3.5596]),
        decimal=3
    )
    return


def test_bspline(fit_model):
    np.random.seed(42)
    input_data = 10 * np.random.rand(30)
    y = (
        2.0 * input_data ** 2 +
        7 +
        10 * np.random.randn(len(input_data))
    )

    grid = np.arange(0, 11, 2.0)
    order = 2
    N = len(grid) + order - 2
    sigma = 10 ** 2
    formula = gammy.BSpline1d(
        grid,
        order=order,
        prior=(np.zeros(N), np.identity(N) / sigma),
        extrapolate=True
    )(x)
    model = fit_model(formula, gammy.numpy.Delta(0.0136602))(input_data, y)
    assert_almost_equal(
        model.predict(input_data[[6, 2, 11, 7, 23]]),
        np.array([11.4905, 112.6615, 172.8875, 145.6188, 27.7447]),
        decimal=1
    )
    return
