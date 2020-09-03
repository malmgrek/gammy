import pytest

import numpy as np
from numpy.testing import (
    assert_array_equal,
    assert_allclose,
    assert_equal
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


def assert_nodes_equal(a, b):
    """Assert that two bp.expfamily nodes coincide

    """
    assert_arrays_equal(a.u, b.u)
    assert_arrays_equal(a.phi, b.phi)
    assert_array_equal(a.f, b.f)
    assert_array_equal(a.g, b.g)
    assert a.observed == b.observed
    return


@pytest.mark.parametrize("data,expected", [
    (
        polynomial(),
        {
            "__len__": 2,
            "theta_marginals": [np.array([7.]), np.array([[49.00071654]])],
            "mean_theta": [np.array([7.]), np.array([2.])],
            "covariance_theta": np.array([
                [ 0.0007166, -0.0003583 ],
                [-0.0003583,  0.00026276]
            ]),
            "inv_mean_tau": 0.000500374,
            "theta_marginal": [np.array([2.]), np.array([4.00026276])],
            "predict": 7 * np.array([0., .2, .4, .6, .8, 1.]) + 2,
            "predict_variance": (
                np.array([2., 3.4, 4.8, 6.2, 7.6, 9.]),
                np.array([
                    0.00076313, 0.00064847, 0.00059114,
                    0.00059114, 0.00064847, 0.00076313
                ])
            ),
            "predict_variance_theta": (
                np.array([2., 3.4, 4.8, 6.2, 7.6, 9.]),
                np.array([
                    2.62755089e-04, 1.48098323e-04, 9.07699398e-05,
                    9.07699398e-05, 1.48098323e-04, 2.62755089e-04
                ])
            ),
            "predict_marginals": [
                np.array([0., 1.4, 2.8, 4.2, 5.6, 7.]),
                np.array([2., 2., 2., 2., 2., 2.])
            ],
            "predict_variance_marginals": [
                (
                    np.array([0., 1.4, 2.8, 4.2, 5.6, 7.]),
                    np.array([
                        0.00000000e+00, 2.86641915e-05, 1.14656766e-04,
                        2.57977723e-04, 4.58627064e-04, 7.16604787e-04
                    ])
                ),
                (
                    np.array([2., 2., 2., 2., 2., 2.]),
                    np.array([
                        0.00026276, 0.00026276, 0.00026276,
                        0.00026276, 0.00026276, 0.00026276
                    ])
                )
            ],
            "marginal_residuals": [
                np.array([
                    -1.98260874e-09, 1.40000000e+00, 2.80000000e+00,
                    4.20000000e+00, 5.60000000e+00, 7.00000000e+00
                ]),
                np.array([2., 2., 2., 2., 2., 2.])
            ]
        },
    )
])
def test_bayesian_gam(data, expected):

    (input_data, y, formula) = data
    model = gammy.BayesianGAM(formula).fit(input_data, y)

    assert len(model) == expected["__len__"]

    assert_allclose(
        model.theta_marginals[0].get_moments(),
        expected["theta_marginals"]
    )

    assert_allclose(model.mean_theta, expected["mean_theta"])

    assert_allclose(
        model.covariance_theta,
        expected["covariance_theta"],
        atol=1e-8
    )

    assert_allclose(
        model.inv_mean_tau,
        expected["inv_mean_tau"],
        atol=1e-8
    )

    assert_allclose(
        model.theta_marginal(1).get_moments(),
        expected["theta_marginal"]
    )
    assert_allclose(
        model.theta_marginal(1).get_moments(),
        model.theta_marginals[1].get_moments()
    )

    assert_allclose(model.predict(input_data), expected["predict"])

    assert_allclose(
        model.predict_variance(input_data),
        expected["predict_variance"],
        atol=1e-8
    )

    assert_allclose(
        model.predict_variance_theta(input_data),
        expected["predict_variance_theta"],
        atol=1e-8
    )

    assert_allclose(
        model.predict_marginals(input_data),
        expected["predict_marginals"],
        atol=1e-8
    )

    assert_allclose(
        model.predict_variance_marginals(input_data),
        expected["predict_variance_marginals"],
        atol=1e-8
    )

    assert_allclose(
        model.predict_marginal(input_data, 0),
        expected["predict_marginals"][0],
        atol=1e-8
    )
    assert_allclose(
        model.predict_marginal(input_data, 1),
        expected["predict_marginals"][1],
        atol=1e-8
    )

    assert_allclose(
        model.predict_variance_marginal(input_data, 0),
        expected["predict_variance_marginals"][0],
        atol=1e-8
    )
    assert_allclose(
        model.predict_variance_marginal(input_data, 1),
        expected["predict_variance_marginals"][1],
        atol=1e-8
    )

    assert_allclose(
        model.marginal_residuals(input_data, y),
        expected["marginal_residuals"],
        atol=1e-10
    )

    assert_allclose(
        model.marginal_residual(input_data, y, 0),
        expected["marginal_residuals"][0]
    )
    assert_allclose(
        model.marginal_residual(input_data, y, 1),
        expected["marginal_residuals"][1]
    )

    return


@pytest.mark.parametrize("data", [
    polynomial(), gp()
])
def test_mutable(data):
    """Currently ``BayesianGAM`` object is mutated when fitted

    """
    (input_data, y, formula) = data
    model_prefit = gammy.BayesianGAM(formula)
    model_fitted = model_prefit.fit(input_data, y)
    assert_arrays_equal(
        model_prefit.mean_theta,  # This changes as a side-effect
        model_fitted.mean_theta
    )
    return


@pytest.mark.parametrize("data", [
    polynomial(), gp()
])
def test_fit_unique(data):
    """Check that fit gives same result both times

    """
    (input_data, y, formula) = data
    model_1 = gammy.BayesianGAM(formula).fit(input_data, y)
    model_2 = gammy.BayesianGAM(formula).fit(input_data, y)
    assert_arrays_equal(
        model_1.mean_theta,
        model_2.mean_theta
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
