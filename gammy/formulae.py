"""Formula type definition and constructors

.. autosummary::
   :toctree:

   Formula
   design_matrix
   Flatten
   Sum
   Kron
   create_from_kernel1d
   ExpSquared1d
   ExpSineSquared1d
   RationalQuadratic1d
   WhiteNoise1d
   OrnsteinUhlenbeck1d
   Scalar
   ReLU
   FlippedReLU
   BSpline1d

"""


from __future__ import annotations
from typing import Callable, List, Tuple

import numpy as np
import scipy as sp
from scipy import interpolate

from gammy import utils
from gammy.arraymapper import ArrayMapper
from gammy.utils import listmap, rlift_basis


def design_matrix(input_data: np.ndarray, basis: List[Callable]):
    """Assemble the design matrix for basis function regression

    """
    return np.hstack([
        f(input_data).reshape(-1, 1) for f in basis
    ])


class Formula():
    """Basis manipulation and design matrix creator

    Parameters
    ----------
    bases : List[Callable]
        Each element is a list of basis functions and corresponds to a term
        in the additive model formula
    prior : Tuple[np.ndarray]
        Mean and precision matrix of the Gaussian prior distribution

    Example
    -------
    Manipulation and operations between formulae

    """

    def __init__(self, bases, prior):
        self.bases = bases
        self.prior = prior
        return


    def __add__(self, other) -> Formula:
        """Addition

        """
        return Formula(
            bases=self.bases + other.bases,
            prior=utils.concat_gaussians([self.prior, other.prior])
        )

    def __mul__(self, input_map: ArrayMapper) -> Formula:
        """Multiplication

        """
        return Formula(
            bases=[
                listmap(
                    lambda f: lambda t: f(t) * input_map(t)
                )(basis) for basis in self.bases
            ],
            prior=self.prior
        )

    def __len__(self) -> int:
        """Number of terms in the formula

        """
        return len(self.bases)

    def __call__(self, *input_maps) -> Formula:
        # TODO: Transform basis
        return Formula(
            bases=[
                rlift_basis(f, m) for (f, m) in zip(self.bases, input_maps)
            ],
            prior=self.prior
        )

    def build_Xi(self, input_data, i) -> np.ndarray:
        """A column block of the design matrix

        """
        return design_matrix(input_data, self.bases[i])

    def build_Xs(self, input_data: np.ndarray):
        """All column blocks as list

        """
        return [
            self.build_Xi(input_data, i) for i, _ in enumerate(self.bases)
        ]

    def build_X(self, input_data: np.ndarray):
        """Design matrix

        """
        return np.hstack(self.build_Xs(input_data))


#
# Operations between formulae
#


def Flatten(formula, prior=None) -> Formula:
    """Flatten the bases of a given formula

    Parameters
    ----------
    formula : Formula
        Flattened formula with a nested list of bases
    prior : Tuple[np.ndarray]
        Prior of the final formula

    In terms of bases: ``[[f1, f2], [g1, g2, g3]] => [[f1, f2, g1, g2, g3]]``

    """
    return Formula(
        bases=[utils.flatten(formula.bases)],
        prior=formula.prior if prior is None else prior
    )


def Sum(formulae, prior=None) -> Formula:
    """Sum (i.e. concatenate) many formulae

    Parameters
    ----------
    formulae : List[Formula]
        Formulas to concatenate
    prior : Tuple[np.ndarray]
        Prior mean and covariance for concatenated formula

    Bases: ([[f1, f2], [g1, g2]], [[h1]]) => [[f1, f2], [g1, g2], [h1]]

    NOTE: `Sum` and `Flatten` are different!

    """
    priors = [formula.prior for formula in formulae]
    return Formula(
        bases=utils.flatten([formula.bases for formula in formulae]),
        prior=utils.concat_gaussians(priors) if prior is None else prior
    )


def Kron(a, b) -> Formula:
    """Tensor product of two Formula bases

    Parameters
    ----------
    a : Formula
    b : Formula

    Non-commutative!

    Let ``u, v`` be eigenvectors of matrices ``A, B``, respectively. Then
    ``u ⊗ v`` is an eigenvector of ``A ⊗ B`` and ``λμ`` is the corresponding
    eigenvalue.

    """
    # NOTE: This is somewhat experimental. The bases must correspond to
    #       "zero-mean" r.v.. Then Kronecker product of covariances
    #       corresponds to the product r.v. of independent r.v.'s.
    #       Check the formula of variance of product of independent r.v.'s.

    # In the same order as in a Kronecker product
    gen = (
        (f, g) for f in utils.flatten(a.bases) for g in utils.flatten(b.bases)
    )

    # Outer product of bases
    basis = listmap(
        lambda funcs: lambda t: funcs[0](t) * funcs[1](t)
    )(gen)

    # Kronecker product of prior means and covariances
    return Formula(
        bases=[basis],
        prior=(
            np.kron(a.prior[0], b.prior[0]),
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
            bases=[mu_basis + basis],
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
    return Formula(bases=[basis], prior=prior)


def Line(prior: Tuple[np.ndarray]=(0, 1e-6)) -> Formula:
    """Straight line (through origin) formula

    """
    basis = [lambda t: t]
    return Formula(bases=[basis], prior=prior)


def Function(function: Callable, prior: Tuple[np.ndarray]) -> Formula:
    """Construct a formula from a single function

    """
    basis = [function]
    return Formula(bases=[basis], prior=prior)


def ReLU(grid: np.ndarray, prior: Tuple[np.ndarray]=None) -> Formula:
    """Rectified linear unit shaped basis

    """
    relus = listmap(lambda c: lambda t: (t > c) * (c - t))(grid[1:-1])
    prior = (
        (np.zeros(len(grid) - 2), np.identity(len(grid) - 2))
        if not prior else prior
    )
    return Formula(bases=[relus], prior=prior)


def FlippedReLU(grid: np.ndarray, prior: Tuple[np.ndarray]=None) -> Formula:
    """Mirrored ReLU basis

    """
    relus = listmap(lambda c: lambda t: (t < c) * (c - t))(grid[1:-1])
    prior = (
        (np.zeros(len(grid) - 2), np.identity(len(grid) - 2))
        if not prior else prior
    )
    return Formula(bases=[relus], prior=prior)


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
    """B-spline basis on a fixed grid

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
        (np.zeros(len(basis)), np.identity(len(basis)))
        if prior is None else prior
    )
    return Formula(
        bases=[mu_basis + basis],
        prior=prior if mu_hyper is None else utils.concat_gaussians(
            [mu_hyper, prior]
        )
    )
