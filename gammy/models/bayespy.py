"""BayesPy engine

"""


import json
from typing import Callable, List, Tuple

import bayespy as bp
import h5py
import numpy as np

import gammy
from gammy import utils
from gammy.formulae import Formula


def create_gaussian_theta(formula: Formula):
    return bp.nodes.Gaussian(*formula.prior)


class GAM:
    """Generalized additive model with BayesPy backend

    Parameters
    ----------
    formula : gammy.formulae.Formula
        Formula object containing the terms and prior
    theta : bp.nodes.Gaussian
        Model parameters vector
    tau : bp.nodes.Gamma
        Observation noise precision (inverse variance)

    Currently tau is fixed to Gamma distribution, i.e., it is not
    possible to manually define the noise level. Note though that one can
    set tight values for `α, β` in `Gamma(α, β)`, recalling that `mean = α / β`
    and `variance = α / β ** 2`. The upside is that by estimating the noise
    level, one gets a nice prediction uncertainty estimate.

    Currently Gaussian requirement deeply built in. Tau being Gamma
    implies, by conjugacy, that theta must be Gaussian.

    Does not support scalar valued Gaussian r.v.. Could be implemented
    using GaussianARD but this would require a lot of refactoring for such a
    small feature -- after all one can define an auxiliary bias term with a
    very tight prior.

    TODO: Statistics for basis function evaluations at grid points.

    FIXME: BayesPy fit fails with the following example:

        .. code-block:: python

           gammy.bayespy.GAM(
               gammy.Scalar()
           ).fit(np.array([0]), np.array([1]))

    """

    def __init__(self, formula, tau=None, theta=None) -> None:
        # NOTE: Pitfall here: setting default value e.g. tau=bp.nodes.Gamma()
        #       would ruin everything because of mutability

        self.formula = formula
        """Model formula"""

        self.tau = tau if tau is not None else bp.nodes.Gamma(1e-3, 1e-3)
        """Node for additive noise precision"""

        self.theta = (
            theta if theta is not None else create_gaussian_theta(formula)
        )
        """Node for model parameters"""

    def __len__(self) -> int:
        """Number of model parameters

        """
        return len(sum(self.formula.terms, []))

    @property
    def theta_marginals(self) -> List[bp.nodes.Gaussian]:
        """Nodes for the basis specific marginal distributions

        """
        # TODO: Test that the marginal distributions are correct
        u = self.theta.get_moments()
        mus = utils.unflatten(u[0], self.formula.terms)
        covs = utils.extract_diag_blocks(
            utils.solve_covariance(u),
            self.formula.terms
        )
        return [
            bp.nodes.Gaussian(mu=mu, Lambda=bp.utils.linalg.inv(cov))
            for (mu, cov) in zip(mus, covs)
        ]

    @property
    def mean_theta(self) -> List[np.ndarray]:
        """Mean estimate of model parameters

        Posterior if model is fitted, otherwise prior.

        """
        return utils.listmap(np.array)(
            utils.unflatten(
                self.theta.get_moments()[0],
                self.formula.terms
            )
        )

    @property
    def covariance_theta(self) -> np.ndarray:
        """Covariance estimate of model parameters

        """
        return utils.solve_covariance(self.theta.get_moments())

    @property
    def inv_mean_tau(self) -> np.ndarray:
        """Additive observation noise variance estimate

        """
        return 1 / self.tau.get_moments()[0]

    def theta_marginal(self, i: int) -> bp.nodes.Gaussian:
        """Extract marginal distribution for a specific term

        """
        u = self.theta.get_moments()
        mus = utils.unflatten(u[0], self.formula.terms)
        covs = utils.extract_diag_blocks(
            utils.solve_covariance(u),
            self.formula.terms
        )
        return bp.nodes.Gaussian(
            mu=mus[i],
            Lambda=bp.utils.linalg.inv(covs[i])
        )

    def fit(self, input_data, y, repeat=1000, verbose=False, **kwargs) -> "GAM":
        """Update BayesPy nodes and construct a GAM predictor

        Parameters
        ----------
        input_data : np.ndarray
            Input data
        y : np.ndarray
            Observations
        repeat : int
            BayesPy allowed repetitions in variational Bayes learning
        verbose : bool
            BayesPy logging

        WARNING: Currently mutates the original object's ``theta`` and ``tau``.

        An option to "reset" the original object to prior is to use the method
        ``initialize_from_prior()`` of BayesPy nodes.

        """
        X = self.formula.design_matrix(input_data)
        F = bp.nodes.SumMultiply("i,i", self.theta, X)
        Y = bp.nodes.GaussianARD(F, self.tau)
        Y.observe(y)
        Q = bp.inference.VB(Y, self.theta, self.tau)
        Q.update(repeat=repeat, verbose=verbose, **kwargs)
        return self

    def predict(self, input_data) -> np.ndarray:
        """Calculate mean of posterior predictive at inputs

        Parameters
        ----------
        input_data : np.ndarray

        """
        X = self.formula.design_matrix(input_data)
        return np.dot(X, np.hstack(self.mean_theta))

    def predict_variance(self, input_data) -> Tuple[np.ndarray]:
        """Predict mean and variance

        Parameters
        ----------
        input_data : np.ndarray

        """
        X = self.formula.design_matrix(input_data)
        F = bp.nodes.SumMultiply("i,i", self.theta, X)
        u = F.get_moments()
        return (u[0], np.diag(utils.solve_covariance(u)) + self.inv_mean_tau)

    def predict_variance_theta(self, input_data) -> Tuple[np.ndarray]:
        """Predict observations with variance from model parameters

        Parameters
        ----------
        input_data : np.ndarray

        """
        X = self.formula.design_matrix(input_data)
        F = bp.nodes.SumMultiply("i,i", self.theta, X)
        # Ensuring correct moments
        #
        # F = F._ensure_moments(
        #     F, bp.inference.vmp.nodes.gaussian.GaussianMoments, ndim=0
        # )
        #
        # NOTE: See also bp.plot.plot_gaussian how std can be calculated
        u = F.get_moments()
        return (u[0], np.diag(utils.solve_covariance(u)))

    def predict_marginals(self, input_data) -> List[np.ndarray]:
        """Predict all terms separately

        Parameters
        ----------
        input_data : np.ndarray

        """
        Xs = [
            self.formula.design_matrix(input_data, i)
            for i in range(len(self.formula))
        ]
        return [np.dot(X, c) for (X, c) in zip(Xs, self.mean_theta)]

    def predict_variance_marginals(self, input_data) -> List[Tuple[np.ndarray]]:
        """Predict variance (theta) for marginal parameter distributions

        Parameters
        ----------
        input_data : np.ndarray

        NOTE: Analogous to self.predict_variance_theta but for marginal
        distributions. Adding observation noise does not make sense as we don't
        know how it is splitted among the model terms.

        """
        Xs = [
            self.formula.design_matrix(input_data, i)
            for i in range(len(self.formula))
        ]
        Fs = [
            bp.nodes.SumMultiply("i,i", theta, X)
            for (X, theta) in zip(Xs, self.theta_marginals)
        ]
        mus = [np.dot(X, c) for (X, c) in zip(Xs, self.mean_theta)]
        sigmas = [np.diag(utils.solve_covariance(F.get_moments())) for F in Fs]
        return list(zip(mus, sigmas))

    def predict_marginal(self, input_data, i: int) -> np.ndarray:
        """Predict a term separately

        Parameters
        ----------
        input_data : np.ndarray

        """
        X = self.formula.design_matrix(input_data, i)
        return np.dot(X, self.mean_theta[i])

    def predict_variance_marginal(
            self,
            input_data,
            i: int
    ) -> Tuple[np.ndarray]:
        """Predict marginal distributions means and variances

        Parameters
        ----------
        input_data : np.ndarray

        """
        # Not refactored with predict_marginal for perf reasons
        X = self.formula.design_matrix(input_data, i)
        F = bp.nodes.SumMultiply("i,i", self.theta_marginal(i), X)
        mu = np.dot(X, self.mean_theta[i])
        sigma = np.diag(utils.solve_covariance(F.get_moments()))
        return (mu, sigma)

    def marginal_residuals(self, input_data, y) -> List[np.ndarray]:
        """Marginal (partial) residuals

        Parameters
        ----------
        input_data : np.ndarray
            Input data
        y : np.ndarray
            Observations

        """
        mus = self.predict_marginals(input_data)
        return [
            y - np.sum(mus[:i] + mus[i + 1:], axis=0)
            for i in range(len(mus))
        ]

    def marginal_residual(self, input_data, y, i: int) -> np.ndarray:
        """Calculate marginal residual for a given term

        Parameters
        ----------
        input_data : np.ndarray

        """
        # Not refactored with marginal_residuals for perf reasons
        mus = self.predict_marginals(input_data)
        return y - np.sum(mus[:i] + mus[i + 1:], axis=0)

    def save(self, filepath: str) -> None:
        """Save the model to disk

        Supported file formats: JSON and HDF5

        """
        file_ext = filepath.split(".")[-1]
        if file_ext in ("h5", "hdf5"):
            with h5py.File(filepath, "w") as h5f:
                group = h5f.create_group("nodes")
                self.tau._save(group.create_group("tau"))
                self.theta._save(group.create_group("theta"))
        elif file_ext == "json":
            with open(filepath, "w+") as jsonf:
                json.dump(
                    obj={
                        "theta": utils.jsonify(self.theta),
                        "tau": utils.jsonify(self.tau)
                    },
                    fp=jsonf
                )
        else:
            raise ValueError(f"Unknown file type: {file_ext}")

    def load(self, filepath: str, **kwargs) -> "GAM":
        """Load model from a file on disk

        """
        file_ext = filepath.split(".")[-1]
        if file_ext in ("h5", "hdf5"):
            with h5py.File(filepath, "r") as h5f:
                tau = self.tau
                tau._load(h5f["nodes"]["tau"])
                theta = create_gaussian_theta(self.formula)
                theta._load(h5f["nodes"]["theta"])
                return GAM(
                    formula=self.formula,
                    tau=tau,
                    theta=theta
                )
        elif file_ext == "json":
            with open(filepath, "r") as jsonf:
                raw = json.load(jsonf)
                tau = utils.set_from_json(raw["tau"], self.tau)
                theta = utils.set_from_json(raw["theta"], self.theta)
        else:
            raise ValueError("Unknown file type: {0}".format(file_ext))
        return GAM(formula=self.formula, tau=tau, theta=theta)


def LinearModel(n_features, prior=None, **kwargs):
    prior = (
        (np.zeros(n_features), 1e-6 * np.identity(n_features)) if prior is None
        else prior
    )
    basis = [(lambda t: t) for i in range(n_features)]
    formula = Formula(terms=[basis], prior=prior)
    return GAM(formula, **kwargs)
