import pytest

import numpy as np
from numpy.testing import assert_array_equal

import gammy
from gammy import utils
from gammy.arraymapper import x


def polynomial():
    """Dummy polynomial model and data

    """
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


def gp():
    """Dummy GP model and data

    """
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
    return (input_data, y, formula)


assert_arrays_equal = utils.compose(
    utils.listmap(lambda xs: assert_array_equal(*xs)),
    list,
    zip
)


def assert_nodes_equal(a, b):
    """Assert that two bp.expfamily nodes coincide

    """
    assert_arrays_equal(a.u, b.u)
    assert_arrays_equal(a.phi, b.phi)
    assert_array_equal(a.f, b.f)
    assert_array_equal(a.g, b.g)
    assert a.observed == b.observed
    return


@pytest.mark.parametrize("data", [
    polynomial(), gp()
])
def test_mutable(data):
    """Currently ``BayesianGAM`` object is mutated when fitted

    FIXME: Make immutable.

    """
    (input_data, y, formula) = data
    model_prefit = gammy.BayesianGAM(formula)
    model_fitted = model_prefit.fit(input_data, y)
    assert_arrays_equal(
        model_prefit.mean_theta,
        model_fitted.mean_theta
    )
    return


@pytest.mark.parametrize("filename", [
    "test.json", "test.hdf5"
])
@pytest.mark.parametrize("data", [
    polynomial(), gp()
])
def test_serialize(tmpdir, filename, data):
    p = tmpdir.mkdir("test").join(filename)
    (input_data, y, formula) = data
    model = gammy.BayesianGAM(formula).fit(input_data, y)
    model.save(p.strpath)
    loaded = gammy.BayesianGAM(formula).load(p.strpath)
    assert_nodes_equal(model.theta, loaded.theta)
    assert_nodes_equal(model.tau, loaded.tau)
    return
