import pytest

import numpy as np
from numpy import testing

from gammy import utils
from gammy.arraymapper import ArrayMapper, lift


np.random.seed(42)
data = np.random.randn(42, 8)


@pytest.mark.parametrize("op", [
    (lambda _, y: y),
    (lambda x, _: x[:, 2]),
    (lambda x, y: x + y),
    (lambda x, y: x - y),
    (lambda x, y: x * y),
    (lambda x, y: x / y),
    (lambda x, _: x ** 2),
    (lambda x, _: -x)
])
def test_arithmetic(op):

    def function(t):
        return t + 42

    x = ArrayMapper()
    y = ArrayMapper(function)
    testing.assert_almost_equal(
        op(x, y)(data),
        op(data, function(data)),
        decimal=8
    )
    return


def test_lift():

    def function(t):
        return t ** 2

    def f(t):
        return t - 2

    x = ArrayMapper(function)
    testing.assert_almost_equal(
        lift(f)(x)(data),
        utils.compose(f, function)(data),
        decimal=8
    )
    return


def test_ravel():
    x = ArrayMapper()
    testing.assert_array_equal(
        x.ravel()(data),
        data.ravel()
    )


def test_reshape():
    x = ArrayMapper()
    testing.assert_array_equal(
        x.reshape(4, 84)(data),
        data.reshape(4, 84)
    )
