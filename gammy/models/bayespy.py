"""BayesPy based GAM model"""


import json
from typing import (Callable, List, Tuple)

import bayespy as bp
import h5py
import numpy as np

import gammy
from gammy import utils
from gammy.utils import listmap, pipe


def build_gaussian_theta(formula: gammy.formulae.Formula):
    return bp.nodes.Gaussian(*formula.prior)


class BayesianGAM(object):
    """Generalized additive model predictor

    Parameters
    ----------
    formula : gammy.Formula
        Formula object containing the bases and prior
    theta : bp.nodes.Gaussian
        Model parameters vector
    tau : bp.nodes.Gamma
        Observation noise precision

    NOTE: Currently tau is fixed to Gamma distribution, i.e., it is not
    possible to manually define the noise level. Note though that one can
    set tight values for `α, β` in `Gamma(α, β)`, recalling that `mean = α / β`
    and `variance = α / β ** 2`. The upside is that by estimating the noise
    level, one gets a nice prediction uncertainty estimate.

    NOTE: Currently Gaussian requirement deeply built in. Tau being Gamma
    implies, by conjugacy, that theta must be Gaussian.

    NOTE: Does not support scalar valued Gaussian r.v.. Could be implemented
    using GaussianARD but this would require a lot of refactoring for such a
    small feature -- after all one can define an auxiliary bias term with a
    very tight prior.

    TODO: Statistics for basis function evaluations at grid points.

    """

    def __init__(
            self,
            formula,
            tau=None,
            theta=None
    ):
        # NOTE: Pitfall here: setting default value e.g. tau=bp.nodes.Gamma()
        #       would ruin everything because of mutability
        self.formula = formula
        self.tau = tau if tau is not None else bp.nodes.Gamma(1e-3, 1e-3)
        self.theta = (
            theta if theta is not None else build_gaussian_theta(formula)
        )

    def __len__(self) -> int:
        """Number of model parameters

        """
        return len(utils.flatten(self.formula.bases))

    @property
    def theta_marginals(self) -> List:
        """Nodes for the basis specific marginal distributions

        """
        # TODO: Test that the marginal distributions are correct
        mus = utils.unflatten(self.theta.get_moments()[0], self.formula.bases)
        covs = utils.extract_diag_blocks(
            utils.solve_covariance(self.theta),
            self.formula.bases
        )
        return [
            bp.nodes.Gaussian(mu=mu, Lambda=bp.utils.linalg.inv(cov))
            for mu, cov in zip(mus, covs)
        ]

    @property
    def mean_theta(self) -> List:
        """Transforms theta to similarly nested list as bases

        """
        return pipe(
            self.theta.get_moments()[0],
            lambda x: utils.unflatten(x, self.formula.bases),
            listmap(np.array)
        )

    @property
    def covariance_theta(self) -> np.ndarray:
        return utils.solve_covariance(self.theta)

    @property
    def inv_mean_tau(self) -> np.ndarray:
        return 1 / self.tau.get_moments()[0]

    def theta_marginal(self, i: int):
        """Extract marginal distribution for a specific term

        """
        mus = utils.unflatten(self.theta.get_moments()[0], self.formula.bases)
        covs = utils.extract_diag_blocks(
            utils.solve_covariance(self.theta),
            self.formula.bases
        )
        return bp.nodes.Gaussian(
            mu=mus[i],
            Lambda=bp.utils.linalg.inv(covs[i])
        )

    def fit(
        self,
        input_data: np.ndarray,
        y: np.ndarray,
        repeat: int=1000,
        verbose: bool=False,
        **kwargs
    ):
        """Update BayesPy nodes and construct a GAM predictor

        WARNING: Currently mutates the original object's ``theta`` and ``tau``.

        An option to "reset" the original object to prior is to use the method
        ``initialize_from_prior()`` of BayesPy nodes.

        """
        X = self.formula.build_X(input_data)
        F = bp.nodes.SumMultiply("i,i", self.theta, X)
        Y = bp.nodes.GaussianARD(F, self.tau)
        Y.observe(y)
        Q = bp.inference.VB(Y, self.theta, self.tau)
        Q.update(repeat=repeat, verbose=verbose, **kwargs)
        return self

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Predict observations

        Returns
        -------
        np.array
            Mean of posterior predictive distribution

        """
        X = self.formula.build_X(input_data)
        return np.dot(X, np.hstack(self.mean_theta))

    def predict_variance(
            self, input_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict observations with variance

        Returns
        -------
        (μ, σ) : tuple([np.array, np.array])
            ``μ`` is mean of posterior predictive distribution
            ``σ`` is variances of posterior predictive + noise

        """
        X = self.formula.build_X(input_data)
        F = bp.nodes.SumMultiply("i,i", self.theta, X)
        return (
            F.get_moments()[0],
            pipe(F, utils.solve_covariance, np.diag) + self.inv_mean_tau
        )

    def predict_variance_theta(
            self, input_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict observations with variance from model parameters

        Returns
        -------
        (μ, σ) : tuple([np.array, np.array])
            ``μ`` is mean of posterior predictive distribution
            ``σ`` is variances of posterior predictive

        """
        X = self.formula.build_X(input_data)
        F = bp.nodes.SumMultiply("i,i", self.theta, X)
        # Ensuring correct moments
        #
        # F = F._ensure_moments(
        #     F, bp.inference.vmp.nodes.gaussian.GaussianMoments, ndim=0
        # )
        #
        # NOTE: See also bp.plot.plot_gaussian how std can be calculated
        return (
            F.get_moments()[0],
            pipe(F, utils.solve_covariance, np.diag)
        )

    def predict_marginals(self, input_data: np.ndarray) -> List:
        """Predict all terms separately

        """
        Xs = self.formula.build_Xs(input_data)
        return [np.dot(X, c) for X, c in zip(Xs, self.mean_theta)]

    def predict_variance_marginals(self, input_data: np.ndarray) -> List:
        Xs = self.formula.build_Xs(input_data)
        Fs = [
            bp.nodes.SumMultiply("i,i", theta, X)
            for X, theta in zip(Xs, self.theta_marginals)
        ]
        mus = [np.dot(X, c) for X, c in zip(Xs, self.mean_theta)]
        sigmas = [pipe(F, utils.solve_covariance, np.diag) for F in Fs]
        return list(zip(mus, sigmas))

    def predict_marginal(self, input_data: np.ndarray, i: int) -> np.ndarray:
        """Predict a term separately

        """
        X = self.formula.build_Xi(input_data, i=i)
        return np.dot(X, self.mean_theta[i])

    def predict_variance_marginal(
            self,
            input_data: np.ndarray,
            i: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Not refactored with predict_marginal for perf reasons
        X = self.formula.build_Xi(input_data, i=i)
        F = bp.nodes.SumMultiply("i,i", self.theta_marginal(i), X)
        mu = np.dot(X, self.mean_theta[i])
        sigma = pipe(F, utils.solve_covariance, np.diag)
        return (mu, sigma)

    def marginal_residuals(
            self, input_data: np.ndarray, y: np.ndarray
    ) -> List:
        """Marginal (partial) residuals

        """
        marginals = self.predict_variance_marginals(input_data)
        mus = [mu for (mu, _) in marginals]
        return [
            y - np.sum(mus[:i] + mus[i + 1:], axis=0)
            for i in range(len(mus))
        ]

    def marginal_residual(
            self,
            input_data: np.ndarray,
            y: np.ndarray,
            i: int
    ) -> np.ndarray:
        # Not refactored with marginal_residuals for perf reasons
        marginals = self.predict_variance_marginals(input_data)
        mus = [mu for (mu, _) in marginals]
        return y - np.sum(mus[:i] + mus[i + 1:], axis=0)

    def _save_h5(self, group):
        self.tau._save(group.create_group("tau"))
        self.theta._save(group.create_group("theta"))
        return group

    def save(self, filepath: str) -> None:
        """Save the model to disk

        """
        file_ext = filepath.split(".")[-1]
        if file_ext in ("h5", "hdf5"):
            with h5py.File(filepath, "w") as h5f:
                group = h5f.create_group("nodes")
                self._save_h5(group)
        elif file_ext == "json":
            with open(filepath, "w+") as jsonf:
                json.dump(
                    obj=dict(
                        theta=utils.jsonify(self.theta),
                        tau=utils.jsonify(self.tau)
                    ),
                    fp=jsonf
                )
        else:
            raise ValueError("Unknown file type: {0}".format(file_ext))

    def _load_h5(
            self,
            h5f: str,
            build_theta: Callable=build_gaussian_theta
    ):
        tau = self.tau
        tau._load(h5f["nodes"]["tau"])
        theta = build_theta(self.formula)
        theta._load(h5f["nodes"]["theta"])
        return BayesianGAM(
            formula=self.formula,
            tau=tau,
            theta=theta
        )

    def _load_json(self, jsonf: str):
        raw = json.load(jsonf)
        return BayesianGAM(
            formula=self.formula,
            tau=utils.set_from_json(raw["tau"], self.tau),
            theta=utils.set_from_json(raw["theta"], self.theta)
        )

    def load(self, filepath: str, **kwargs):
        """Load model from disk

        """
        file_ext = filepath.split(".")[-1]
        if file_ext in ("h5", "hdf5"):
            with h5py.File(filepath, "r") as h5f:
                return self._load_h5(h5f, **kwargs)
        elif file_ext == "json":
            with open(filepath, "r") as jsonf:
                return self._load_json(jsonf)
        else:
            raise ValueError("Unknown file type: {0}".format(file_ext))
