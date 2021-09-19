"""Miscellaneous utilities

"""


import functools
from typing import (Callable, Dict, Generator, Iterable, List)

import numpy as np
import scipy.interpolate as spi


#
# Functional
# ~~~~~~~~~~
#


def curryish(f: Callable) -> Callable:

    def g(*args, **kwargs):
        return functools.partial(f, *args, **kwargs)

    return g


def compose2(f: Callable, g: Callable) -> Callable:

    def h(*args, **kwargs):
        return f(g(*args, **kwargs))

    return h


def lift(func: Callable) -> Callable:
    """Transforms a function into an operator

    lift :: (a -> b) -> ((c -> a) -> (c -> b))

    NOTE: Could add func's *args and **kwargs as arguments

    """
    return lambda f: compose2(func, f)


def lift2(func: Callable) -> Callable:
    return (
        lambda f, g: (
            lambda *args, **kwargs: func(
                *[f(*args, **kwargs), g(*args, **kwargs)]
            )
        )
    )


def rlift(func: Callable) -> Callable:
    """Lift from right

    """
    return lambda f: compose2(f, func)


def compose(*funcs: Callable) -> Callable:
    return functools.partial(functools.reduce, compose2)(funcs)


def pipe(arg, *funcs: Callable) -> Callable:
    return compose(*funcs[::-1])(arg)


listmap = curryish(compose(list, map))
tuplemap = curryish(compose(tuple, map))
listfilter = curryish(compose(list, filter))
tuplefilter = curryish(compose(tuple, filter))


#
# Iterables
# ~~~~~~~~~
#


def flatten(x: List) -> List:
    """Flatten a list of lists once

    """
    return functools.reduce(lambda cum, this: cum + this, x, [])


def unflatten(x: List, y: List) -> List:
    """Unflatten according to a reference

    Example
    -------

    .. code-block:: python

        unflatten([1, 2, 3], [["a", "b"], ["c"]])
        # [[1, 2], [3]]

    """
    def func(cum, this):
        x_crop, res = cum
        return [
            x_crop[len(this):], res + [x_crop[:len(this)]]
        ]

    return functools.reduce(func, list(y), [list(x), []])[-1]


def extract_diag_blocks(x: np.ndarray, y: List) -> List:
    """Extract diagonal blocks from a matrix according to a reference

    """

    def func(cum, this):
        x_crop, res = cum
        return [
            x_crop[len(this):, len(this):],
            res + [x_crop[:len(this), :len(this)]]
        ]

    return functools.reduce(func, list(y), [x, []])[-1]


def extend_spline_grid(grid: np.ndarray, order: int) -> np.ndarray:
    if order < 1:
        raise ValueError(
            "Spline order = n + 1 where n >= 0 is the polynomial degree"
        )
    return grid if order == 1 else pipe(
        grid,
        lambda x: np.append(
            x, np.diff(x)[-(order - 1):][::-1].cumsum() + x[-1]
        ),
        lambda x: np.append(
            x[0] - np.diff(x)[:(order - 1)].cumsum()[::-1], x
        )
    )


def gen_spline_args_from_grid_ext(
        grid_ext: np.ndarray,
        order: int,
        extrapolate: bool
) -> Generator:
    n = len(grid_ext) - order  # Number of basis functions
    (i_left, i_right) = (
        (1, n - 1) if order == 1 else (order - 1, n - order + 1)
    )
    return (
        (grid_ext[i:i + order + 1],) + (
            (extrapolate, -1) if i < i_left
            else (
                (extrapolate, 1) if i >= i_right else (False, 0)
            )
        )
        for i in range(n)
    )


#
# Basis function generation tools
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# TODO / FIXME: The linear-algebraic stuff below remains unfortunately
# largely untested.
#


def squared_dist(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Squared distance matrix for column array of N-dimensional points

    Example
    -------

    .. code-block:: python

        x = np.array([[0], [1], [2]])
        squared_dist(x, x)
        # array([[0, 1, 4],
        #        [1, 0, 1],
        #        [4, 1, 0]])

    """
    return (
        np.sum(x1 ** 2, 1).reshape(-1, 1) +
        np.sum(x2 ** 2, 1) -
        2 * np.dot(x1, x2.T)
    )


def exp_squared(
        x1: np.ndarray,
        x2: np.ndarray,
        corrlen: float=1.0,
        sigma: float=1.0
) -> np.ndarray:
    return sigma * np.exp(-0.5 / corrlen ** 2 * squared_dist(x1, x2))


def exp_sine_squared(
        x1: np.ndarray,
        x2: np.ndarray,
        corrlen: float=1.0,
        sigma: float=1.0,
        period: float=1.0
) -> np.ndarray:
    return sigma * np.exp(
        -2.0 / corrlen ** 2 * np.sin(
            np.pi * np.sqrt(squared_dist(x1, x2)) / period
        ) ** 2
    )


def rational_quadratic(
        x1: np.ndarray,
        x2: np.ndarray,
        corrlen: float=1.0,
        sigma: float=1.0,
        alpha: float=1.0
) -> np.ndarray:
    return sigma * (
        1 + squared_dist(x1, x2) / 2.0 / alpha / corrlen ** 2
    ) ** -alpha


def ornstein_uhlenbeck(
        x1: np.ndarray,
        x2: np.ndarray,
        corrlen: float=1.0,
        sigma: float=1.0
) -> np.ndarray:
    return sigma * np.exp(
        -np.sqrt(squared_dist(x1, x2)) / corrlen
    )


def white_noise(n_dims: int, sigma: float=1.0) -> np.ndarray:
    return sigma * np.identity(n_dims)


def decompose_covariance(H: np.ndarray, energy: float=1.01) -> np.ndarray:
    """Most important eigenvectors of a symmetric positive-definite square matrix

    Ordered with respect of the descending eigenvalues. Each
    eigenvector scaled with ``sqrt(λ)``.

    Parameters
    ----------
    H : np.ndarray
        Symmetric positive-definite square matrix
    energy : float
        Truncate to eigenvalues that sum up to this proportion of the total
        eigenvalue sum. If absolutelu all eigenvectors are needed, give value
        slightly larger than one.

    Theoretical background
    ----------------------

    Lets expand the solution along the span of the vectors
    :math:`\lambda_n^{1/2} \mathbf{u}_n`, where :math:`\{\lambda_n`,
    :math:`\mathbf{u}_n\}_n` is the eigenbasis of the covariance matrix
    :math:`\Sigma`. Note that the eigenvectors of real symmetric matrices are
    orthonormal. The eigensystem of the precision matrix :math:`\Lambda =
    \Sigma^{-1}` is given by :math:`\{\lambda_n^{-1}`, :math:`\mathbf{u}_n\}_n`.
    Expressing :math:`x = \mu + y_n\lambda_n^{1/2} \mathbf{u}_n`, we get for the
    log-prior

    .. math::

        (x - \mu)^T \Lambda (x - \mu)
        = \lambda_n^{1/2}\mathbf{u}_n^T y_n^2 \lambda_n^{-1/2}\mathbf{u}_n
        = \|\mathbf{y}\|^2.

    In other words, the given transformation for the log-prior is whitening.

    NOTE: In the implementation we use np.linalg.svd instead of np.linalg.eigh
    because the latter sometimes returns slightly negative eigenvalues for
    numerical reasons. In those cases the energy trick doesn't give all
    eigenvectors even if we wanted

    REVIEW: There might be problem with serialization. If there are duplicate
            eigenvalues, then on different machines, the vectors might
            appear in different order.

    """

    #
    # Comparison of np.linalg.eigh and np.linalg.svd
    #
    # (W, V) = np.linalg.eigh(H)
    # (U, S, Vh) = np.linalg.svd(H)
    #
    # Holds up to numerical sanity: V[:, ::-1] == U == Vh.T
    #

    (U, S, Vh) = np.linalg.svd(H)
    crop = (S.cumsum() / S.sum()) <= energy
    return np.dot(U[:, crop], np.sqrt(np.diag(S[crop])))


def interp1d_1darrays(v: np.ndarray, grid: np.ndarray, **kwargs) -> List:
    """Create list of interpolators from a given array

    Parameters
    ----------
    v : np.array
        Each column is a "basis" vector

    """
    return [
        spi.interp1d(grid, v[:, i], **kwargs) for i in range(v.shape[1])
    ]


def rlift_basis(basis: List[Callable], func: Callable) -> List:
    return listmap(rlift(func))(basis)


#
# Files and I/O
# ~~~~~~~~~~~~~
#


def write_to_hdf5(group, data, name):
    try:
        group.create_dataset(name, data=data, compression="gzip")
    except TypeError:
        group.create_dataset(name, data=data)
    except ValueError:
        raise ValueError(f"Could not write {data}")


#
# BayesPy related
# ~~~~~~~~~~~~~~~
#


def concat_gaussians(gaussians: List[Tuple[np.ndarray, np.ndarray]]):
    """Concatenate means and covariances to one Gaussian

    Parameters
    ----------
    gaussians : [(μ1, Λ1), (μ2, Λ2)]

    """
    return (
        np.hstack([g[0] for g in gaussians]),
        sp.linalg.block_diag(*[g[1] for g in gaussians])
    )


def solve_covariance(u: List[np.ndarray]) -> np.ndarray:
    """Solve covariance matrix from moments

    Parameters
    ----------
    u : List[np.ndarray]
        List of moments as defined by the ``get_moments()`` method call
        of a BayesPy node object.

    """
    cov = u[1] - np.outer(u[0], u[0])
    return cov if cov.shape != (1, 1) else np.array(cov.sum())


solve_precision = compose(np.linalg.inv, solve_covariance)


def jsonify(node) -> Dict:
    """Turn a expfamily node into a JSON serializable dict

    """
    return {
        **{
            "u{0}".format(i):
                ui.tolist() for (i, ui) in enumerate(node.u)
        },
        **{
            "observed": node.observed
        },
        **{
            "phi{0}".format(i):
                phii.tolist() for (i, phii) in enumerate(node.phi)
        },
        **{
            "f": node.f.tolist(),
            "g": node.g.tolist()
        }
    }


def set_from_json(raw: Dict, node):
    """Set BayesPy node attributes from JSON

    """
    node.u = [
        np.array(raw["u{0}".format(i)]) for i in range(len(node.u))
    ]
    node.observed = raw["observed"]
    node.phi = [
        np.array(raw["phi{0}".format(i)]) for i in range(len(node.phi))
    ]
    node.f = np.array(raw["f"])
    node.g = np.array(raw["g"])
    return node
