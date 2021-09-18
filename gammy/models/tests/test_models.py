import pytest

import numpy as np
from numpy.testing import (
    assert_array_equal,
    assert_almost_equal
)

import gammy
from gammy import utils
from gammy.arraymapper import x


def polynomial():
    """Dummy polynomial model and data

    """
    input_data = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y = 7 * input_data + 2
    formula = gammy.Scalar((0, 1e-6)) * x + gammy.Scalar((0, 1e-6))
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


@pytest.mark.parametrize("model_data", [
    utils.pipe(
        polynomial(),
        lambda xs: (
            xs[0],
            xs[1],
            gammy.models.bayespy.GAM(formula=xs[2]).fit(
                input_data=xs[0],
                y=xs[1]
            )
        )
    ),
    utils.pipe(
        polynomial(),
        lambda xs: (
            xs[0],
            xs[1],
            gammy.models.numpy.GAM(
                formula=xs[2],
                tau=gammy.models.numpy.Delta(1998.50381764)
            ).fit(
                input_data=xs[0],
                y=xs[1]
            )
        )
    )
])
def test_gam(model_data):
    """Test Numpy and BayesPy-based GAM

    - Check that the interfaces are same
    - Check that numeric results are sufficiently close

    NOTE: Currently we are testing here just the dummy polynomial model. I
    didn't see it necessary copying a large set of data regarding a more complex
    model as that test would be model validation, not testing that the class
    works.

    """
    (input_data, y, model) = model_data

    #
    # model.__len__
    # ~~~~~~~~~~~~~
    #
    assert len(model) == 2

    #
    # model.theta_marginals
    # ~~~~~~~~~~~~~~~~~~~~~
    #
    ans = model.theta_marginals[0].get_moments()
    assert_almost_equal(ans[0], [7], decimal=8)
    assert_almost_equal(ans[1], [[49]], decimal=3)
    ans = model.theta_marginals[1].get_moments()
    assert_almost_equal(ans[0], [2], decimal=8)
    assert_almost_equal(ans[1], [[4]], decimal=3)

    #
    # model.mean_theta
    # ~~~~~~~~~~~~~~~~
    #
    assert_almost_equal(
        model.mean_theta,
        [np.array([7]), np.array([2])],
        decimal=8
    )

    #
    # model.covariance_theta
    # ~~~~~~~~~~~~~~~~~~~~~~
    #
    assert_almost_equal(
        model.covariance_theta,
        np.array([
            [ 0.00071, -0.00035],
            [-0.00035,  0.00026]
        ]),
        decimal=5
    )

    #
    # model.inv_mean_tau
    # ~~~~~~~~~~~~~~~~~~
    #
    assert_almost_equal(
        model.inv_mean_tau,
        0.00050037,
        decimal=8
    )

    #
    # model.theta_marginal
    # ~~~~~~~~~~~~~~~~~~~~
    #
    theta_marginals = model.theta_marginals
    assert_almost_equal(
        [model.theta_marginal(i).get_moments()[0] for i in range(len(model))],
        [theta_marginals[i].get_moments()[0] for i in range(len(model))]
    )

    #
    # model.predict
    # ~~~~~~~~~~~~~
    #
    assert_almost_equal(
        model.predict(input_data),
        [2, 3.4, 4.8, 6.2, 7.6, 9],
        decimal=8
    )

    #
    # model.predict_variance
    # ~~~~~~~~~~~~~~~~~~~~~~
    #
    (y_pred, sigma) = model.predict_variance(input_data)
    assert_almost_equal(y_pred, [2, 3.4, 4.8, 6.2, 7.6, 9], decimal=8)
    assert_almost_equal(
        sigma,
        [0.00076, 0.00065, 0.00059, 0.00059, 0.00064, 0.00076],
        decimal=5
    )

    #
    # model.predict_variance_theta
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    (y_pred, sigma) = model.predict_variance_theta(input_data)
    assert_almost_equal(y_pred, [2, 3.4, 4.8, 6.2, 7.6, 9], decimal=8)
    assert_almost_equal(
        sigma,
        [0.00026, 0.00015, 0.00009, 0.00009, 0.00015, 0.00026],
        decimal=5
    )

    #
    # model.predict_marginals
    # ~~~~~~~~~~~~~~~~~~~~~~~
    #
    marginals = model.predict_marginals(input_data)
    assert_almost_equal(marginals[0], [0, 1.4, 2.8, 4.2, 5.6, 7], decimal=8)
    assert_almost_equal(marginals[1], [2, 2, 2, 2, 2, 2], decimal=8)

    #
    # model.predict_variance_marginals
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    variance_marginals = model.predict_variance_marginals(input_data)
    assert_almost_equal(
        variance_marginals[0][0], [0, 1.4, 2.8, 4.2, 5.6, 7], decimal=8
    )
    assert_almost_equal(
        variance_marginals[0][1],
        [0, 0.000029, 0.00011, 0.00026, 0.00046, 0.00072],
        decimal=5
    )
    assert_almost_equal(
        variance_marginals[1][0], [2, 2, 2, 2, 2, 2], decimal=8
    )
    assert_almost_equal(
        variance_marginals[1][1],
        [0.00026, 0.00026, 0.00026, 0.00026, 0.00026, 0.00026], decimal=5
    )

    #
    # model.predict_marginal
    # ~~~~~~~~~~~~~~~~~~~~~~
    #
    marginals = model.predict_marginals(input_data)
    assert_almost_equal(
        [model.predict_marginal(input_data, i) for i in range(len(model))],
        marginals
    )

    #
    # model.predict_variance_marginal
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    variance_marginals = model.predict_variance_marginals(input_data)
    assert_almost_equal(
        [
            model.predict_variance_marginal(input_data, i)
            for i in range(len(model))
        ],
        variance_marginals,
        decimal=8
    )

    #
    # model.marginal_residuals
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    #
    assert_almost_equal(
        model.marginal_residuals(input_data, y),
        [
            np.array([0, 1.4, 2.8, 4.2, 5.6, 7]),
            np.array([2, 2, 2, 2, 2, 2])
        ],
        decimal=8
    )

    #
    # model.marginal_residual
    # ~~~~~~~~~~~~~~~~~~~~~~~
    #
    marginal_residuals = model.marginal_residuals(input_data, y)
    assert_almost_equal(
        marginal_residuals,
        [model.marginal_residual(input_data, y, i) for i in range(len(model))]
    )
    return


@pytest.mark.parametrize("data", [
    polynomial(), gp()
])
def test_bayespy_mutable(data):
    """Currently ``gammy.models.bayespy.GAM`` object is mutated when fitted

    """
    (input_data, y, formula) = data
    model_prefit = gammy.models.bayespy.GAM(formula)
    model_fitted = model_prefit.fit(input_data, y)
    assert_arrays_equal(
        model_prefit.mean_theta,  # This changes as a side-effect
        model_fitted.mean_theta
    )
    return


@pytest.mark.parametrize("data", [
    polynomial(), gp()
])
def test_numpy_immutable(data):
    (input_data, y, formula) = data
    return


@pytest.mark.parametrize("data", [
    polynomial(), gp()
])
def test_fit_unique(data):
    """Check that fit gives same result both times

    """
    (input_data, y, formula) = data
    model_1 = gammy.models.bayespy.GAM(formula).fit(input_data, y)
    model_2 = gammy.models.bayespy.GAM(formula).fit(input_data, y)
    assert_arrays_equal(
        model_1.mean_theta,
        model_2.mean_theta
    )
    return


def assert_nodes_equal(a, b):
    """Assert that two bp.expfamily nodes coincide

    """
    assert_arrays_equal(a.u, b.u)
    assert_arrays_equal(a.phi, b.phi)
    assert_array_equal(a.f, b.f)
    assert_array_equal(a.g, b.g)
    assert a.observed == b.observed
    return


@pytest.mark.parametrize("filename", [
    "test.json",
    "test.hdf5"
])
@pytest.mark.parametrize("data", [
    polynomial(),
    gp()
])
def test_numpy_serialize(tmpdir, filename, data):
    p = tmpdir.mkdir("test").join(filename)
    (input_data, y, formula) = data
    model = gammy.models.numpy.GAM(
        formula=formula,
        tau=gammy.models.numpy.Delta(666)
    )
    model.save(p.strpath)
    loaded = gammy.models.numpy.GAM(
        formula=formula,
        tau=gammy.models.numpy.Delta(42)
    ).load(p.strpath)
    assert_almost_equal(model.theta.mu, loaded.theta.mu, decimal=8)
    assert_almost_equal(model.theta.Lambda, loaded.theta.Lambda, decimal=8)
    assert_almost_equal(model.tau.mu, loaded.tau.mu, decimal=8)
    return


@pytest.mark.parametrize("filename", [
    "test.json", "test.hdf5"
])
@pytest.mark.parametrize("data", [
    polynomial(), gp()
])
def test_bayespy_serialize(tmpdir, filename, data):
    p = tmpdir.mkdir("test").join(filename)
    (input_data, y, formula) = data
    model = gammy.models.bayespy.GAM(formula).fit(input_data, y)
    model.save(p.strpath)
    loaded = gammy.models.bayespy.GAM(formula).load(p.strpath)
    assert_nodes_equal(model.theta, loaded.theta)
    assert_nodes_equal(model.tau, loaded.tau)
    return
