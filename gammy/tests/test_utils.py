"""Unit tests for utils"""


import pytest

from gammy import utils


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


def test_lift():
    return


def test_rlift():
    return


def test_unflatten():
    return


def test_solve_covariance():
    return
