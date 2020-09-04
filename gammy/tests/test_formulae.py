"""Unit tests for formula type"""


import numpy as np
from numpy.testing import (
    assert_array_equal,
    assert_allclose,
    assert_equal
)
import pytest

import gammy
from gammy import utils
from gammy.arraymapper import x


# TODO: Multivariate with Kron
def multinomial():
    return (
        gammy.Scalar() * x[:, 0] * x[:, 1] + gammy.Scalar(),
        np.vstack(
            (
                np.arange(0, 1.5, 0.5),
                -np.arange(0, 1.5, 0.5)
            )
        ).T
    )


def scalar():
    return (
        gammy.Scalar(),
        np.array([0., 1., 2.])
    )


def line():
    return (
        gammy.Scalar((1, 2)) * x,
        np.array([0., 1., 2.])
    )


def sine():
    return (
        gammy.Function(
            utils.compose(np.sin, lambda t: 0.5 * np.pi * t),
            (0, 1)
        ),
        np.array([0., 1., 2.])
    )


def square():
    return (
        gammy.Scalar() * x ** 2,
        np.array([0., 1., 2.])
    )


def test_concat_gaussians():
    X = (np.ones(2), np.identity(2))
    Y = (np.zeros(3), 2 * np.ones((3, 3)))
    (mean, precision) = gammy.concat_gaussians([X, Y])
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


@pytest.mark.parametrize("xs,expected", [
    (
        multinomial(),
        [
            - np.arange(0, 1.5, 0.5).reshape(-1, 1) ** 2,
            np.ones(3).reshape(-1, 1)
        ]
    )
])
def test_design_matrix(xs, expected):
    (formula, input_data) = xs
    assert_array_equal(
        utils.listmap(
            lambda b: gammy.design_matrix(input_data, b)
        )(formula.bases),
        expected
    )
    return


@pytest.mark.parametrize("xs,ys,input_map,expected", [
    (
        line(),
        sine(),
        x,
        {
            "__add__": np.array([
                [0., 0.],
                [1., 1.],
                [2., 0.]
            ]),
            "__mul__": np.array([0., 1., 4.])[:, None],
            "__len__": 1,
            "__call__": np.array([0., .5, 1.])[:, None],
            "build_Xi": np.array([0., 1., 2.])[:, None],
            "build_X": np.array([0., 1., 2.])[:, None]
        },
    )
])
@pytest.mark.parametrize("require", [
    lambda d, f, o, i, e: assert_allclose(
        (f + o).build_X(d),
        e["__add__"],
        atol=1e-12
    ),
    lambda d, f, o, i, e: assert_array_equal(
        (f * i).build_X(d),
        e["__mul__"]
    ),
    lambda d, f, o, i, e: assert_equal(len(f), e["__len__"]),
    lambda d, f, o, i, e: assert_array_equal(
        f(lambda t: t * .5).build_X(d),
        e["__call__"]
    ),
    lambda d, f, o, i, e: assert_array_equal(
        f.build_X(d),
        e["build_X"]
    ),
    lambda d, f, o, i, e: assert_array_equal(
        (f + o).build_Xi(d, 0),
        e["build_Xi"]
    )
])
def test_formula(xs, ys, input_map, expected, require):
    (formula, input_data) = xs
    (other, _) = ys
    require(input_data, formula, other, input_map, expected)
    return


def test_flatten():
    formula = line()[0] + sine()[0] + multinomial()[0]
    assert all([len(bs) == 1 for bs in formula.bases])
    [bs] = gammy.formulae.Flatten(formula).bases
    assert len(bs) == 4
    return


def test_sum():
    formula = gammy.formulae.Sum([line()[0], sine()[0], multinomial()[0]])
    assert all([len(b) == 1 for b in formula.bases])
    assert_equal(
        formula.prior,
        (
            np.array([1., 0., 0., 0.]),
            np.array([
                [2., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ])
        )
    )
    return


def test_kron():
    input_data = np.array([0., 1., 2., 3.])
    a = gammy.formulae.Flatten(line()[0] + sine()[0])
    b = gammy.formulae.Flatten(line()[0] + square()[0])
    formula = gammy.formulae.Kron(a, b)
    [bs] = formula.bases
    assert len(bs) == 4
    assert_allclose(
        formula.build_X(input_data),
        np.array([
            [0., 0., 0., 0.],
            [1., 1., 1., 1.],
            [4., 8., 0., 0.],
            [9., 27., -3., -9.]
        ]),
        atol=1e-10
    )
    return
