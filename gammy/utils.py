"""Miscellaneous utilities

"""

import functools
from typing import (Callable, Dict, Generator, Iterable, List, Tuple)

import numpy as np
import scipy as sp
from scipy import interpolate


#
# Functional
# ~~~~~~~~~~
#


# TODO: A proper curry
def curryish(f: Callable) -> Callable:
    """Lifted partial application

    """

    def g(*args, **kwargs):
        return functools.partial(f, *args, **kwargs)

    return g


def compose2(f: Callable, g: Callable) -> Callable:
    """Compose two functions

    """

    def h(*args, **kwargs):
        return f(g(*args, **kwargs))

    return h


def lift(func: Callable) -> Callable:
    """Transforms a function into an operator

    ``lift :: (a -> b) -> ((c -> a) -> (c -> b))``

    NOTE: Could add func's *args and **kwargs as arguments

    """
    return lambda f: compose2(func, f)


def rlift(func: Callable) -> Callable:
    """Lift from right

    """
    return lambda f: compose2(f, func)


def compose(*funcs: Callable) -> Callable:
    """Function composition

    """
    return functools.partial(functools.reduce, compose2)(funcs)


def pipe(arg, *funcs: Callable) -> Callable:
    """Piping an object through functions

    """
    return compose(*funcs[::-1])(arg)


listmap = curryish(compose(list, map))
listmap.__doc__ = """Map for lists with partial evaluation

"""

tuplemap = curryish(compose(tuple, map))
tuplemap.__doc__ = """Map for tuples with partial evaluation

"""

listfilter = curryish(compose(list, filter))
listfilter.__doc__ = """Filter for lists with partial evaluation

"""

tuplefilter = curryish(compose(tuple, filter))
tuplefilter.__doc__ = """Filter for tuples with partial evaluation

"""


#
# Iterables
# ~~~~~~~~~
#


def unflatten(x: list, y: list) -> list:
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


def extract_diag_blocks(x: np.ndarray, y: list) -> List[np.ndarray]:
    """Extract diagonal blocks from a matrix according to a reference

    """

    def func(cum, this):
        x_crop, res = cum
        return [
            x_crop[len(this):, len(this):],
            res + [x_crop[:len(this), :len(this)]]
        ]

    return functools.reduce(func, list(y), [x, []])[-1]


def extend_spline_grid(grid, order: int) -> np.ndarray:
    """Grid extension for higher order splines

    """
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


def gen_spline_args_from_grid_ext(grid_ext: np.ndarray, order: int, extrapolate: bool) -> Generator:
    """Spline arguments generator from extended grid

    Parameters
    ----------
    grid_ext : np.ndarray
        Extended grid
    order : int
        Order of the splines
    extrapolate : bool
        Allow smooth(ish) extrapolation

    """
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


def squared_dist(x1, x2) -> np.ndarray:
    """Squared distance matrix for column array of N-dimensional points

    Parameters
    ----------
    x1 : np.ndarray
        1-D Column array
    x2 : np.ndarray
        1-D Column array

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


def exp_squared(x1, x2, corrlen=1.0, sigma=1.0) -> np.ndarray:
    """Exponential squared kernel function

    """
    return sigma * np.exp(-0.5 / corrlen ** 2 * squared_dist(x1, x2))


def exp_sine_squared(x1, x2, corrlen=1.0, sigma=1.0, period=1.0) -> np.ndarray:
    """Exponential sine squared kernel function

    """
    return sigma * np.exp(
        -2.0 / corrlen ** 2 * np.sin(
            np.pi * np.sqrt(squared_dist(x1, x2)) / period
        ) ** 2
    )


def rational_quadratic(x1, x2, corrlen=1.0, sigma=1.0, alpha=1.0) -> np.ndarray:
    """Rational quadratic kernel function

    """
    return sigma * (
        1 + squared_dist(x1, x2) / 2.0 / alpha / corrlen ** 2
    ) ** -alpha


def ornstein_uhlenbeck(x1, x2, corrlen=1.0, sigma=1.0) -> np.ndarray:
    """Ornstein-Uhlenbeck kernel function

    """
    return sigma * np.exp(
        -np.sqrt(squared_dist(x1, x2)) / corrlen
    )


def white_noise(n_dims: int, sigma=1.0, **unused) -> np.ndarray:
    """White noise kernel function

    """
    return sigma * np.identity(n_dims)


def decompose_covariance(H, energy: float=1.01) -> np.ndarray:
    """Most important eigenvectors of a symmetric positive-definite square matrix

    Parameters
    ----------
    H : np.ndarray
        Symmetric positive-definite square matrix
    energy : float
        Truncate to eigenvalues that sum up to this proportion of the total
        eigenvalue sum. If absolutelu all eigenvectors are needed, give value
        slightly larger than one.

    Ordered with respect of the descending eigenvalues. Each
    eigenvector scaled with ``sqrt(λ)``. For theoretical justification,
    see the section on Gaussian Processes in the package documentation.

    NOTE: In the implementation we use np.linalg.svd instead of np.linalg.eigh
    because the latter sometimes returns slightly negative eigenvalues for
    numerical reasons. In those cases the energy trick doesn't give all
    eigenvectors even if we wanted

    REVIEW: There might be problem with serialization. If there are duplicate
    eigenvalues, then on different machines, the vectors might appear in
    different order.

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


def interp_arrays1d(v, grid, **kwargs) -> List:
    """Create list of interpolators from a given array

    Parameters
    ----------
    v : np.array
        Each column is a "basis" vector
    grid : np.ndarray
        Interpolation grid

    """
    return [
        interpolate.interp1d(grid, v[:, i], **kwargs)
        for i in range(v.shape[1])
    ]


#
# Files and I/O
# ~~~~~~~~~~~~~
#


def write_to_hdf5(group, data, name):
    """Add data to HDF5 handler

    """
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


def concat_gaussians(gaussians):
    """Concatenate means and covariances to one Gaussian

    Parameters
    ----------
    gaussians : List[Tuple[np.ndarray]]
        List of mean-precision tuples of each Gaussian

    """
    return (
        np.hstack([g[0] for g in gaussians]),
        sp.linalg.block_diag(*[g[1] for g in gaussians])
    )


def solve_covariance(u) -> np.ndarray:
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
solve_precision.__doc__ = """Solve precision matrix from moments

"""


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


def set_from_json(raw: dict, node):
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


def peaks(x, y):
    """The MATLAB function

    """
    return (
        3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) -
        10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) -
        1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
    )
