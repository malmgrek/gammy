"""Unit tests for utils"""


import bayespy as bp
import numpy as np
from numpy import array_equal
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal
)
import pytest

from gammy import utils


np.random.seed(42)


@pytest.mark.parametrize("f,g,xs", [
    (
        lambda x, y: x + y,
        lambda x: (lambda y: x + y),
        [(0, 0), (1, 1), (1, 2), (6, 42)]
    )
])
def test_curryish(f, g, xs):
    assert all([
        f(*args) == g(args[0])(args[1]) for args in xs
    ])
    return


@pytest.mark.parametrize("fs,xs,g", [
    (
        [
            lambda x: x ** 2,
            lambda x: x - 1,
            lambda x, y: x + y
        ],
        [
            (1, 2), (42, 666), (-10, 6), (213543.123, -1724)
        ],
        lambda x, y: x ** 2 + 2 * x * y + y ** 2 - 2 * x - 2 * y + 1

    )
])
def test_compose(fs, xs, g):
    assert all([
        utils.compose(*fs)(*args) == g(*args) for args in xs
    ])
    return


@pytest.mark.parametrize("x,y,expected", [
    (
        [[], 1, None],
        [["a", "b"], [1]],
        [[[], 1], [None]]
    ),
    (
        [], [], []
    ),
    (
        [1, 2, 3], [], []
    ),
    (
        [1], [[()]], [[1]]
    )
])
def test_unflatten(x, y, expected):
    assert utils.unflatten(x, y) == expected
    return


@pytest.mark.parametrize("x,y,expected", [
    (
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]),
        [
            np.array([
                [1, 1],
                [1, 1]
            ]),
            np.array([[1]])
        ],
        [
            np.array([
                [1, 2],
                [4, 5]
            ]),
            np.array([[9]])
        ]
    ),
    (
        np.array([
            [1, 2],
            [3, 4]
        ]),
        [
            np.array([
                [np.nan, np.nan],
                [np.nan, np.inf]
            ]),
            np.array([])
        ],
        [
            np.array([
                [1, 2],
                [3, 4]
            ])
        ]
    )
])
def test_extract_diag_blocks(x, y, expected):
    assert all([
        array_equal(*args) for args in zip(
            expected,
            utils.extract_diag_blocks(x, y)
        )
    ])
    return


@pytest.mark.parametrize("mu,Sigma", [
    (
        np.random.rand(23),
        utils.pipe(
            np.random.randn(23, 23),
            lambda A: np.dot(A.T, A)
        )
    )
])
def test_solve_covariance(mu, Sigma):
    node = bp.nodes.Gaussian(mu, np.linalg.inv(Sigma))
    assert_almost_equal(
        utils.solve_covariance(node.get_moments()),
        Sigma,
        decimal=8
    )
    return


@pytest.mark.parametrize("order,expected", [
    (
        1, np.array([0, 1, 2, 3]),
    ),
    (
        2, np.array([-1, 0, 1, 2, 3, 4])
    ),
    (
        3, np.array([-2, -1, 0, 1, 2, 3, 4, 5])
    )
])
def test_extend_spline_grid(order, expected):
    grid = np.array([0, 1, 2, 3])
    assert_array_equal(
        utils.extend_spline_grid(grid, order),
        expected
    )
    return


@pytest.mark.parametrize("x1,x2,expected", [
    (
        np.array([[0, 0], [1, 2]]),
        np.array([[1, 1], [2, 2], [3, 1]]),
        np.array([
            [2, 8, 10],
            [1, 1, 5]
        ])
    ),
    (
        np.array([[0], [1], [2], [3]]),
        np.array([[0], [1], [2], [3]]),
        np.array([
            [0, 1, 4, 9],
            [1, 0, 1, 4],
            [4, 1, 0, 1],
            [9, 4, 1, 0]
        ])
    )
])
def test_squared_dist(x1, x2, expected):
    assert_array_equal(utils.squared_dist(x1, x2), expected)
    return


def test_exp_squared():
    x1 = np.array([[0], [1], [2], [3]])
    x2 = np.array([[0], [1], [2], [3]])
    sigma = 2
    corrlen = 2
    expected = 2 * np.exp(
        -0.5 * np.array([
            [0, 1, 4, 9],
            [1, 0, 1, 4],
            [4, 1, 0, 1],
            [9, 4, 1, 0]
        ]) / 4
    )
    assert_almost_equal(
        utils.exp_squared(x1, x2, sigma=sigma, corrlen=corrlen),
        expected,
        decimal=12
    )
    return


def test_exp_sine_squared():
    x1 = np.array([[0], [1], [2], [3]])
    x2 = np.array([[0], [1], [2], [3]])
    sigma = 2
    corrlen = 2
    period = 2
    expected = 2 * np.exp(
        -2.0 * np.sin(
            np.pi * np.array([
                [0, 1, 2, 3],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [3, 2, 1, 0]
            ]) / 2
        ) ** 2 / 4
    )
    assert_almost_equal(
        utils.exp_sine_squared(
            x1, x2, sigma=sigma, period=period, corrlen=corrlen
        ),
        expected,
        decimal=12
    )
    return


def test_rational_quadratic():
    x1 = np.array([[0], [1], [2], [3]])
    x2 = np.array([[0], [1], [2], [3]])
    sigma = 2
    corrlen = 2
    alpha = 1.5
    expected = 2 * np.exp(
        -1.5 * np.log(
            1.0 + 0.5 * np.array([
                [0, 1, 4, 9],
                [1, 0, 1, 4],
                [4, 1, 0, 1],
                [9, 4, 1, 0]
            ]) / 4 / 1.5
        )
    )
    assert_almost_equal(
        utils.rational_quadratic(
            x1, x2, sigma=sigma, corrlen=corrlen, alpha=alpha
        ),
        expected,
        decimal=12
    )
    return


def test_ornstein_uhlenbeck():
    x1 = np.array([[0], [1], [2], [3]])
    x2 = np.array([[0], [1], [2], [3]])
    sigma = 2
    corrlen = 2
    expected = 2 * np.exp(
        -np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0]
        ]) / 2
    )
    assert_almost_equal(
        utils.ornstein_uhlenbeck(x1, x2, sigma=sigma, corrlen=corrlen),
        expected,
        decimal=12
    )
    return


def test_white_noise():
    sigma = 2
    n_dims = 42
    expected = 2 * np.eye(n_dims)
    assert_array_equal(
        utils.white_noise(sigma=sigma, n_dims=n_dims),
        expected
    )
    return


def test_decompose_covariance():
    x1 = np.arange(0, 1, 0.01).reshape(-1, 1)
    H = utils.exp_squared(x1, x1, sigma=1, corrlen=1)
    Uh = utils.decompose_covariance(H, energy=1.01)
    assert_almost_equal(Uh[42, 0], 0.9958898213927759, decimal=12)
    assert_almost_equal(Uh[66, 1], -0.1625888269228036, decimal=12)
    assert_almost_equal(Uh[71, 3], 0.0085321682733649, decimal=12)
    assert_almost_equal(Uh.shape, H.shape)
    return


def test_concat_gaussians():
    X = (np.ones(2), np.identity(2))
    Y = (np.zeros(3), 2 * np.ones((3, 3)))
    (mean, precision) = utils.concat_gaussians([X, Y])
    assert_array_equal(
        mean, np.array([1., 1., 0., 0., 0])
    )
    assert_array_equal(
        precision,
        np.array([
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 2., 2., 2.],
            [0., 0., 2., 2., 2.],
            [0., 0., 2., 2., 2.]
        ])
    )
