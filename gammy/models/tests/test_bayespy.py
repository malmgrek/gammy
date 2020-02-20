import pytest

import numpy as np
from numpy.testing import assert_array_equal

import gammy
from gammy.arraymapper import x


def dummy_data():
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
    return (input_data, y, formula)


def test_mutable():
    """Currently ``BayesianGAM`` object is mutated when fitted

    FIXME: Make immutable.

    """
    (input_data, y, formula) = dummy_data()
    model_prefit = gammy.BayesianGAM(formula)
    model_fitted = model_prefit.fit(input_data, y)
    assert np.array_equal(
        model_prefit.mean_theta,
        model_fitted.mean_theta
    )
    return
