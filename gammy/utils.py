import functools

import bayespy as bp
import numpy as np
import scipy.interpolate as spi


#
# Function manipulation
#


def curryish(f):

    def g(*args, **kwargs):
        return functools.partial(f, *args, **kwargs)

    return g


def compose2(f, g):

    def h(*args, **kwargs):
        return f(g(*args, **kwargs))

    return h


def lift(func):
    # Could add func's *args, **kwargs here
    return lambda f: compose2(func, f)


def lift2(func):
    return (
        lambda f, g: (
            lambda *args, **kwargs: func(
                *[f(*args, **kwargs), g(*args, **kwargs)]
            )
        )
    )


def rlift(func):
    return lambda f: compose2(f, func)


def compose(*funcs):
    return functools.partial(functools.reduce, compose2)(funcs)


def pipe(arg, *funcs):
    return compose(*funcs[::-1])(arg)


listmap = curryish(compose(list, map))
tuplemap = curryish(compose(tuple, map))
listfilter = curryish(compose(list, filter))
tuplefilter = curryish(compose(tuple, filter))



#
# Iterable manipulations
#


def flatten(x):
    """Flatten a list of lists once

    """
    return functools.reduce(lambda cum, this: cum + this, x, [])


def unflatten(x, y):
    """Unflatten according to a reference

    """
    def func(cum, this):
        x_crop, res = cum
        return [
            x_crop[len(this):], res + [x_crop[:len(this)]]
        ]

    return functools.reduce(func, list(y), [list(x), []])[-1]


def extract_diag_blocks(x, y):
    """Extract diagonal blocks from a matrix according to a reference

    """

    def func(cum, this):
        x_crop, res = cum
        return [
            x_crop[len(this):, len(this):],
            res + [x_crop[:len(this), :len(this)]]
        ]

    return functools.reduce(func, list(y), [x, []])[-1]


#
# Basis function generation tools
#


def squared_dist(X1, X2):
    """Squared distance matrix for column array of N-dimensional points

    Example
    -------

    .. code-block:: python

        X = np.array([[0], [1], [2]])
        squared_dist(X, X)
        # array([[0, 1, 4],
        #        [1, 0, 1],
        #        [4, 1, 0]])

    """
    return (
        np.sum(X1 ** 2, 1).reshape(-1, 1) +
        np.sum(X2 ** 2, 1) -
        2 * np.dot(X1, X2.T)
    )


def exp_squared(X1, X2, l=1.0, sigma=1.0):
    return sigma * np.exp(-0.5 / l ** 2 * squared_dist(X1, X2))


def exp_sine_squared(X1, X2, l=1.0, sigma=1.0, period=1.0):
    return sigma * np.exp(
        -2.0 / l ** 2 * np.sin(
            np.pi * np.sqrt(squared_dist(X1, X2)) / period
        ) ** 2
    )


def rational_quadratic(X1, X2, l=1.0, sigma=1.0, alpha=1.0):
    return sigma * (
        1 + squared_dist(X1, X2) / 2.0 / alpha / l ** 2
    ) ** -alpha


def white_noise(n_dims, sigma=1.0):
    return sigma * np.identity(n_dims)


def scaled_principal_eigvecsh(H, energy=0.99):
    """Most important eigenvectors of a hermitian matrix

    Descending order with respect of the corresponding eigenvalues. Each
    vector scaled with ``sqrt(λ)``.

    Lets expand the solution as :math:`\lambda_n^{1/2} \mathbf{u}_n`,
    where :math:`\{\lambda_n`, :math:`\mathbf{u}_n\}_n` is the eigenbasis
    of the covariance matrix :math:`\Sigma`. Note that then the eigenbasis of
    the precision matrix :math:`\Lambda = \Sigma^{-1}` is given by
    :math:`\{\lambda_n^{-1}`, :math:`\mathbf{u}_n\}_n`. Expressing
    :math:`x = \mu + w_n\lambda_n^{1/2} \mathbf{u}_n`, we get for the
    logarithm of the prior distribution

    .. math::

        (x - \mu)^T \Lambda (x - \mu)
        = \lambda_n^{1/2}\mathbf{u}_n^T \lambda_n^{-1/2}\mathbf{u}_n^T
        = \|\mathbf{w}\|^2


    """
    w, v = pipe(
        np.linalg.eigh(H),
        lambda x: (x[0][::-1], x[1][:, ::-1])
    )
    crop = (w.cumsum() / w.sum()) <= energy
    return pipe(
        v,
        lambda x: np.dot(x[:, crop], np.sqrt(np.diag(w[crop])))
    )


def interp1d_1darrays(v, grid, **kwargs):
    """Create list of interpolators from a given array

    Parameters
    ----------
    v : np.array
        Each column is a "basis" vector

    """
    return [
        spi.interp1d(
            grid, v[:, i], **kwargs) for i in range(v.shape[1]
        )
    ]


def rlift_basis(basis, func):
    return listmap(rlift(func))(basis)


#
# BayesPy related
#


def update(formula, input_data, y, tau, **kwargs):
    """Updates BayesPy nodes

    """
    (theta, F) = formula.build_nodes(input_data)
    Y = bp.nodes.GaussianARD(F, tau)
    Y.observe(y)
    Q = bp.inference.VB(Y, theta, tau)
    Q.update(**kwargs)
    return (theta, F, tau, Y)


def solve_covariance(node):
    # FIXME: Works only for Gaussian?
    u = node.get_moments()
    cov = u[1] - np.outer(u[0], u[0])
    return cov if cov.shape != (1, 1) else np.array(cov.sum())
