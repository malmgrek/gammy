"""This module defines the formula type"""


from typing import List

import attr
import numpy as np
import scipy as sp
from scipy import interpolate

from gammy import utils
from gammy.utils import listmap, rlift_basis


def concat_gaussians(gaussians: list):
    # gaussians = [(μ1, Λ1), (μ2, Λ2)]
    return (
        np.hstack([g[0] for g in gaussians]),
        sp.linalg.block_diag(*[g[1] for g in gaussians])
    )


def design_matrix(input_data, basis):
    return np.hstack([
        f(input_data).reshape(-1, 1) for f in basis
    ])


@attr.s(frozen=True)
class Formula():
    """Basis manipulation and design matrix creator

    Parameters
    ----------
    bases : list
        Each element is a list of basis functions and corresponds to a term
        in the additive model formula
    prior : tuple
        Mean and precision matrix of the Gaussian prior distribution

    Example
    -------
    Manipulation and operations between formulae

    """

    bases = attr.ib()
    prior = attr.ib()

    def __add__(self, other):
        return Formula(
            bases=self.bases + other.bases,
            prior=concat_gaussians([self.prior, other.prior])
        )

    def __mul__(self, input_map):
        return Formula(
            bases=[
                listmap(
                    lambda f: lambda t: f(t) * input_map(t)
                )(basis) for basis in self.bases
            ],
            prior=self.prior
        )

    def __len__(self):
        """Number of terms in the formula

        """
        return len(self.bases)

    def __call__(self, *input_maps):
        # TODO: Transform basis
        return Formula(
            bases=[
                rlift_basis(f, m) for (f, m) in zip(self.bases, input_maps)
            ],
            prior=self.prior
        )

    def build_Xi(self, input_data, i):
        """A column block of the design matrix

        """
        return design_matrix(input_data, self.bases[i])

    def build_Xs(self, input_data):
        """All column blocks as list

        """
        return [
            self.build_Xi(input_data, i) for i, _ in enumerate(self.bases)
        ]

    def build_X(self, input_data):
        """Design matrix

        """
        return np.hstack(self.build_Xs(input_data))


#
# Operations between formulae
#


def Flatten(formula: Formula, prior=None):
    """Flatten the bases of a given formula

    Bases: [[f1, f2], [g1, g2, g3]] => [[f1, f2, g1, g2, g3]]

    """
    return Formula(
        bases=[utils.flatten(formula.bases)],
        prior=formula.prior if prior is None else prior
    )


def Sum(formulae: List[Formula], prior=None):
    """Sum (i.e. concatenate) many formulae

    Bases: ([[f1, f2], [g1, g2]], [[h1]]) => [[f1, f2], [g1, g2], [h1]]

    NOTE: Differs from `Flatten`

    """
    priors = [formula.prior for formula in formulae]
    return Formula(
        bases=utils.flatten([formula.bases for formula in formulae]),
        prior=concat_gaussians(priors) if prior is None else prior
    )


def Kron(a, b):
    """Tensor product of two Formula bases

    Non-commutative!

    Parameters
    ----------
    a : Formula
    b : Formula

    Returns
    -------
    Formula

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


def ExpSquared1d(
    grid, corrlen, sigma, prior=None, mu_basis=None, mu_hyper=None, energy=0.99
):
    """Squared exponential model term

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
    mu_basis = [] if mu_basis is None else mu_basis
    basis = utils.interp1d_1darrays(
        utils.scaled_principal_eigvecsh(
            utils.exp_squared(
                X1=grid.reshape(-1, 1),
                X2=grid.reshape(-1, 1),
                corrlen=corrlen,
                sigma=sigma
            ),
            energy=energy
        ),
        grid=grid,
        fill_value="extrapolate"
    )
    # Default prior is white noise
    prior = (
        (np.zeros(len(basis)), np.identity(len(basis)))
        if prior is None else prior
    )
    return Formula(
        bases=[mu_basis + basis],
        prior=prior if mu_hyper is None else concat_gaussians(
            [mu_hyper, prior]
        )
    )


def ExpSineSquared1d(
    grid, corrlen, sigma, period,
    prior=None, mu_basis=None, mu_hyper=None, energy=0.99
):
    mu_basis = [] if mu_basis is None else mu_basis
    basis = utils.interp1d_1darrays(
        utils.scaled_principal_eigvecsh(
            utils.exp_sine_squared(
                X1=grid.reshape(-1, 1),
                X2=grid.reshape(-1, 1),
                corrlen=corrlen,
                sigma=sigma,
                period=period
            ),
            energy=energy
        ),
        grid=grid,
        fill_value="extrapolate"
    )
    # Default prior is white noise
    prior = (
        (np.zeros(len(basis)), np.identity(len(basis)))
        if prior is None else prior
    )
    return Formula(
        bases=[mu_basis + basis],
        prior=prior if mu_hyper is None else concat_gaussians(
            [mu_hyper, prior]
        )
    )


def WhiteNoise1d(
    grid, sigma, prior=None, mu_basis=None, mu_hyper=None, energy=1.0
):
    mu_basis = [] if mu_basis is None else mu_basis
    basis = utils.interp1d_1darrays(
        utils.scaled_principal_eigvecsh(
            utils.white_noise(n_dims=len(grid), sigma=sigma),
            energy=energy
        ),
        grid=grid,
        fill_value="extrapolate"
    )
    # Default prior is white noise
    prior = (
        (np.zeros(len(basis)), np.identity(len(basis)))
        if prior is None else prior
    )
    return Formula(
        bases=[mu_basis + basis],
        prior=prior if mu_hyper is None else concat_gaussians(
            [mu_hyper, prior]
        )
    )


def Scalar(prior=(0, 1)):
    basis = [lambda t: np.ones(len(t))]
    return Formula(bases=[basis], prior=prior)


def Line(prior=(0, 1)):
    basis = [lambda t: t]
    return Formula(bases=[basis], prior=prior)


def Function(function, prior):
    basis = [function]
    return Formula(bases=[basis], prior=prior)


def ReLU(grid, prior=None):
    relus = listmap(lambda c: lambda t: (t > c) * (c - t))(grid[1:-1])
    prior = (
        (np.zeros(len(grid) - 2), np.identity(len(grid) - 2))
        if not prior else prior
    )
    return Formula(bases=[relus], prior=prior)


def FlippedReLU(grid, prior=None):
    relus = listmap(lambda c: lambda t: (t < c) * (c - t))(grid[1:-1])
    prior = (
        (np.zeros(len(grid) - 2), np.identity(len(grid) - 2))
        if not prior else prior
    )
    return Formula(bases=[relus], prior=prior)


def TanH():
    raise NotImplementedError


def Gaussian1d():
    raise NotImplementedError


def BSpline1d(grid, order=3, extrapolate=True,
              prior=None, mu_basis=None, mu_hyper=None):
    """B-spline basis on a fixed grid

    Parameters
    ----------

    order : int
        Order of the spline function. Polynomial degree is ``order - 1``
    extrapolate : bool
        Extrapolate outside of the grid using basis functions "touching" the
        endpoints

    Number of spline basis functions is always ``N = len(grid) + order - 2``

    """
    # TODO: Verify that this doesn't break when scaling the grid
    #       (extrapolation + damping)

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
        prior=prior if mu_hyper is None else concat_gaussians(
            [mu_hyper, prior]
        )
    )
