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


@pytest.mark.parametrize("deepcopy", [
    True, False
])
def test_deepcopy(deepcopy):
    (input_data, y, formula) = dummy_data()
    model_prefit = gammy.BayesianGAM(formula)
    model_fitted = model_prefit.fit(input_data, y, deepcopy=deepcopy)
    assert np.array_equal(
        model_prefit.mean_theta,
        model_fitted.mean_theta
    ) != deepcopy
    return


def test_repeating_fits():
    (input_data, y, formula) = dummy_data()
    model0 = gammy.BayesianGAM(formula).fit(input_data, y, deepcopy=True)
    model1 = model0.fit(input_data, y, deepcopy=True)
    model2 = model0.fit(input_data, y, deepcopy=True)
    assert_array_equal(model1.mean_theta, model2.mean_theta)
    return
