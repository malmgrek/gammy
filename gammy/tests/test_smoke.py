import numpy as np
from numpy.testing import assert_allclose

import gammy
from gammy.arraymapper import x


def test_polynomial():
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
    model = gammy.BayesianGAM(formula).fit(input_data, y)
    assert_allclose(
        model.predict(input_data[[0, 3, 6, 9, 12]]),
        np.array([
            52.57112684, 104.28633909, 14.48274839, 136.98584437, 180.09263955
        ])
    )
    return


def test_gp():
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
        l=1.0,
        sigma=1.0,
        period=2 * np.pi,
        energy=0.99
    )(x[:, 0]) * x[:, 1] + gammy.Scalar(prior=(0, 1e-6))
    model = gammy.BayesianGAM(formula).fit(input_data, y)
    assert_allclose(
        model.predict(input_data[[1, 42, 11, 26, 31]]),
        np.array([1.78918855, 1.87107355, 1.30149328, 1.29221874, 1.31596102])
    )
    return


def test_kron():
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
            l=0.5,
            sigma=4.0,
            energy=0.99
        )(x[:, 0]),
        gammy.ExpSquared1d(
            np.arange(-3, 3, 0.1),
            l=0.5,
            sigma=4.0,
            energy=0.99
        )(x[:, 1])
    ) + gammy.Scalar(prior=(0, 1e-6))
    model = gammy.BayesianGAM(formula).fit(input_data, y)
    assert_allclose(
        model.predict(input_data[[1, 5, 12, 19], :]),
        np.array([3.78593199, 3.56403323, 3.73910833, 3.55960657])
    )
    return


def test_bspline():
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
    model = gammy.BayesianGAM(formula).fit(input_data, y)
    assert_allclose(
        model.predict(input_data[[6, 2, 11, 7, 23]]),
        np.array([
            11.49053477, 112.66157312, 172.88750407, 145.61888784, 27.74474919
        ])
    )
    return
