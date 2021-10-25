"""Formula type definition and constructors

"""


from typing import Callable, List, Tuple, Iterable

import numpy as np
import scipy as sp
from scipy import interpolate

from gammy import utils
from gammy.arraymapper import ArrayMapper
from gammy.utils import listmap, rlift


class Formula:
    """Basis manipulation and design matrix creator

    Parameters
    ----------
    terms : List[Callable]
        Each element is a list of basis functions and corresponds to a term
        in the additive model formula
    prior : Tuple[np.ndarray]
        Mean and precision matrix of the Gaussian prior distribution

    """

    def __init__(self, terms, prior):

        self.terms = terms
        """List of basis functions """

        self.prior = prior
        """Prior mean and precision"""

        return


    def __add__(self, other) -> "Formula":
        """Addition of formulae

        Parameters
        ----------
        other : Formula

        """
        return Formula(
            terms=self.terms + other.terms,
            prior=utils.concat_gaussians([self.prior, other.prior])
        )

    def __mul__(self, input_map) -> "Formula":
        """Multiplication

        Parameters
        ----------
        input_map : ArrayMapper

        """
        return Formula(
            terms=[
                listmap(
                    lambda f: lambda t: f(t) * input_map(t)
                )(basis) for basis in self.terms
            ],
            prior=self.prior
        )

    def __len__(self) -> int:
        """Number of terms in the formula

        """
        return len(self.terms)

    def __call__(self, *input_maps) -> "Formula":
        """Make the object callable

        Parameters
        ----------
        input_maps : Iterable[ArrayMapper]

        """
        # TODO: Transform basis
        return Formula(
            terms=[
                listmap(rlift(input_map))(basis)
                for (basis, input_map) in zip(self.terms, input_maps)
            ],
            prior=self.prior
        )

    def design_matrix(self, input_data, i: int=None):
        # If one term is asked for, give it. Otherwise use all terms.
        fs = sum(self.terms, []) if i is None else self.terms[i]
        return np.hstack([f(input_data).reshape(-1, 1) for f in fs])


#
# Operations between formulae
#


def Flatten(formula, prior=None) -> Formula:
    """Flatten the terms of a given formula

    Optionally override prior

    Parameters
    ----------
    formula : Formula
        Flattened formula with a nested list of terms
    prior : Tuple[np.ndarray]
        Prior of the final formula

    In terms of terms: ``[[f1, f2], [g1, g2, g3]] => [[f1, f2, g1, g2, g3]]``

    """
    return Formula(
        terms=[sum(formula.terms, [])],
        prior=formula.prior if prior is None else prior
    )


def Sum(formulae, prior=None) -> Formula:
    """Sum (concatenate) many formulae

    Parameters
    ----------
    formulae : List[Formula]
        Formulas to concatenate
    prior : Tuple[np.ndarray]
        Prior mean and covariance for concatenated formula

    Theoretical example:

    .. code-block:: text

        ([[f1, f2], [g1, g2]], [[h1]]) => [[f1, f2], [g1, g2], [h1]]

    NOTE: :class:`Sum` and :class:`Flatten` are different!

    """
    priors = [formula.prior for formula in formulae]
    return Formula(
        terms=sum([formula.terms for formula in formulae], []),
        prior=utils.concat_gaussians(priors) if prior is None else prior
    )


def Kron(a, b) -> Formula:
    """Tensor product of two Formula terms

    Parameters
    ----------
    a : Formula
        Left input
    b : Formula
        Right input

    Non-commutative!

    Let ``u, v`` be eigenvectors of matrices ``A, B``, respectively. Then
    ``u ⊗ v`` is an eigenvector of ``A ⊗ B`` and ``λμ`` is the corresponding
    eigenvalue.

    """
    # NOTE: This is somewhat experimental. The terms must correspond to
    #       "zero-mean" r.v.. Then Kronecker product of covariances
    #       corresponds to the product r.v. of independent r.v.'s.
    #       Check the formula of variance of product of independent r.v.'s.

    # TODO / FIXME: Don't flatten a and b.
    gen = (
        # Careful! Must be same order as in a Kronecker product.
        (f, g) for f in sum(a.terms, []) for g in sum(b.terms, [])
    )

    # Outer product of terms
    basis = listmap(
        lambda funcs: lambda t: funcs[0](t) * funcs[1](t)
    )(gen)

    # Kronecker product of prior means and covariances
    return Formula(
        terms=[basis],
        prior=(
            np.kron(a.prior[0], b.prior[0]),
            # Although we kron-multiply precision matrices here (inverse
            # of covariance), the order of inputs doesn't flip because
            # (A ⊗ B) ^ -1 = (A ^ -1) ⊗ (B ^ -1)
            np.kron(a.prior[1], b.prior[1])
        )
    )


#
# Custom formulae collection
#


def create_from_kernel1d(kernel: Callable) -> Callable:
    """Create formula from bivariate GP kernel function

    """

    def _Formula(
            grid: np.ndarray,
            prior: Tuple[np.ndarray]=None,
            mu_basis: List[Callable]=None,
            mu_hyper: Tuple[np.ndarray]=None,
            energy: float=.99,
            **kernel_kwargs
    ) -> Formula:

        mu_basis = [] if mu_basis is None else mu_basis
        basis = utils.interp_arrays1d(
            utils.decompose_covariance(
                kernel(
                    x1=grid.reshape(-1, 1),
                    x2=grid.reshape(-1, 1),
                    **kernel_kwargs
                ),
                energy=energy
            ),
            grid=grid,
            fill_value="extrapolate"
        )

        # Default prior is white noise for the problem with
        # the constructed basis functions
        prior = (
            (np.zeros(len(basis)), np.identity(len(basis)))
            if prior is None else prior
        )

        return Formula(
            terms=[mu_basis + basis],
            prior=prior if mu_hyper is None else utils.concat_gaussians(
                [mu_hyper, prior]
            )
        )

    return _Formula


def ExpSquared1d(
        grid,
        corrlen,
        sigma,
        prior=None,
        mu_basis=None,
        mu_hyper=None,
        energy=0.99
) -> Formula:
    """Squared exponential kernel formula

    Example
    -------

    .. code-block:: python

        formula = ExpSquared1d(
            grid=np.arange(-25, 35, 1.0),
            l=5.0,
            sigma=1.0,
            mu_basis=[lambda t: t],
            mu_hyper=(0, 1e-6)
        )

    """
    kernel_kwargs = {
        "corrlen": corrlen,
        "sigma": sigma
    }
    _Formula = create_from_kernel1d(utils.exp_squared)
    return _Formula(
        grid=grid,
        prior=prior,
        mu_basis=mu_basis,
        mu_hyper=mu_hyper,
        energy=energy,
        **kernel_kwargs
    )


def ExpSineSquared1d(
        grid,
        corrlen,
        sigma,
        period,
        prior=None,
        mu_basis=None,
        mu_hyper=None,
        energy=0.99
) -> Formula:
    """Squared sine exponential kernel formula for periodic terms

    Parameters
    ----------
    grid : np.ndarray
        Discretization grid
    corrlen : float
        Correlation length
    sigma : float
        Variance
    period : float
        Period
    prior : Tuple[np.ndarray]
        Prior mean and precision matrix
    mu_basis : List[Callable]
        Basis for estimating the mean hyperparameter
    mu_hyper : Tuple[np.ndarray]
        Hyperprior mean and precision matrix
    energy : float
        Eigenvalue-weighted proportion of eigenvectors to consider in
        truncation

    """
    kernel_kwargs = {
        "corrlen": corrlen,
        "sigma": sigma,
        "period": period
    }
    _Formula = create_from_kernel1d(utils.exp_sine_squared)
    return _Formula(
        grid=grid,
        prior=prior,
        mu_basis=mu_basis,
        mu_hyper=mu_hyper,
        energy=energy,
        **kernel_kwargs
    )


def RationalQuadratic1d(
        grid,
        corrlen,
        sigma,
        alpha,
        prior=None,
        mu_basis=None,
        mu_hyper=None,
        energy=0.99
) -> Formula:
    """Rational quadratic kernel formula

    """
    kernel_kwargs = {
        "corrlen": corrlen,
        "sigma": sigma,
        "alpha": alpha
    }
    _Formula = create_from_kernel1d(utils.rational_quadratic)
    return _Formula(
        grid=grid,
        prior=prior,
        mu_basis=mu_basis,
        mu_hyper=mu_hyper,
        energy=energy,
        **kernel_kwargs
    )


def WhiteNoise1d(
        grid,
        sigma,
        prior=None,
        mu_basis=None,
        mu_hyper=None,
        energy=1.0,
) -> Formula:
    """White noise kernel formula

    """
    kernel_kwargs = {
        "n_dims": len(grid),
        "sigma": sigma
    }
    _Formula = create_from_kernel1d(utils.white_noise)
    return _Formula(
        grid=grid,
        prior=prior,
        mu_basis=mu_basis,
        mu_hyper=mu_hyper,
        energy=energy,
        **kernel_kwargs
    )


def OrnsteinUhlenbeck1d(
        grid,
        corrlen,
        sigma,
        prior=None,
        mu_basis=None,
        mu_hyper=None,
        energy=0.99
) -> Formula:
    """Ornstein-Uhlenbeck kernel formula

    """
    kernel_kwargs = {
        "corrlen": corrlen,
        "sigma": sigma,
    }
    _Formula = create_from_kernel1d(utils.ornstein_uhlenbeck)
    return _Formula(
        grid=grid,
        prior=prior,
        mu_basis=mu_basis,
        mu_hyper=mu_hyper,
        energy=energy,
        **kernel_kwargs
    )


def Scalar(prior: Tuple[np.ndarray]=(0, 1e-6)) -> Formula:
    """Scalar formula

    """
    basis = [lambda t: np.ones(len(t))]
    return Formula(terms=[basis], prior=prior)


def Function(function: Callable, prior: Tuple[np.ndarray]) -> Formula:
    """Construct a formula from a single function

    """
    basis = [function]
    return Formula(terms=[basis], prior=prior)


def Polynomial(degree, prior=None):

    def monomial(p):
        return lambda t: t ** p

    basis = [monomial(n) for n in range(degree + 1)]
    prior = (np.zeros(degree + 1), 1e-6 * np.identity(degree + 1))
    return Formula(terms=[basis], prior=prior)


def ReLU(grid: np.ndarray, prior: Tuple[np.ndarray]=None) -> Formula:
    """Rectified linear unit shaped basis

    """
    relus = listmap(lambda c: lambda t: (t > c) * (c - t))(grid[1:-1])
    prior = (
        (np.zeros(len(grid) - 2), np.identity(len(grid) - 2))
        if not prior else prior
    )
    return Formula(terms=[relus], prior=prior)


def FlippedReLU(grid: np.ndarray, prior: Tuple[np.ndarray]=None) -> Formula:
    """Mirrored ReLU basis

    """
    relus = listmap(lambda c: lambda t: (t < c) * (c - t))(grid[1:-1])
    prior = (
        (np.zeros(len(grid) - 2), np.identity(len(grid) - 2))
        if not prior else prior
    )
    return Formula(terms=[relus], prior=prior)


def TanH() -> Formula:
    raise NotImplementedError


def Gaussian1d() -> Formula:
    raise NotImplementedError


def BSpline1d(
        grid,
        order=3,
        extrapolate=True,
        prior=None,
        mu_basis=None,
        mu_hyper=None
) -> Formula:
    """B-spline basis on a fixed one-dimensional grid

    Parameters
    ----------
    grid : np.ndarray
        Discretization grid
    order : int
        Order of the spline function. Polynomial degree is ``order - 1``
    extrapolate : bool
        Extrapolate outside of the grid using basis functions "touching" the
        endpoints
    prior : Tuple[np.ndarray]
        Prior mean and precision matrix
    mu_basis : List[Callable]
        Basis for estimating the mean hyperparameter
    mu_hyper : Tuple[np.ndarray]
        Hyperprior mean and precision matrix

    Number of spline basis functions is always ``N = len(grid) + order - 2``

    TODO: Verify that this doesn't break when scaling the grid
          (extrapolation + damping)

    """

    mu_basis = [] if mu_basis is None else mu_basis
    grid_ext = utils.extend_spline_grid(grid, order)

    def build_basis_element(spline_arg):

        (knots, extrapolate, loc) = spline_arg

        def right_damp(t):
            return t > knots[-1]

        def left_damp(t):
            return knots[0] > t

        def element(t):
            sp_element = interpolate.BSpline.basis_element(
                knots,
                extrapolate if loc in (-1, 1) else False
            )
            return sp_element(t) if loc == 0 else (
                sp_element(t) * right_damp(t) if loc == -1 else
                sp_element(t) * left_damp(t)
            )

        return utils.compose2(np.nan_to_num, element)

    basis = listmap(build_basis_element)(
        utils.gen_spline_args_from_grid_ext(grid_ext, order, extrapolate)
    )

    # Default prior is white noise
    prior = (
        (np.zeros(len(basis)), 1e-6 * np.identity(len(basis)))
        if prior is None else prior
    )
    return Formula(
        terms=[mu_basis + basis],
        prior=prior if mu_hyper is None else utils.concat_gaussians(
            [mu_hyper, prior]
        )
    )
