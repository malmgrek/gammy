"""Numpy engine

"""

import json
from typing import List, Tuple

import h5py
import numpy as np

import gammy
from gammy import utils
from gammy.formulae import Formula


class Gaussian:
    """Moment calculator for a Gaussian distribution

    Parameters
    ----------
    mu : np.ndarray
        Mean value
    Lambda : np.ndarray
        Precision matrix (inverse of covariance)

    """

    def __init__(self, mu, Lambda) -> None:

        self.mu = mu
        """Mean vector"""

        self.Lambda = Lambda
        """Precision matrix"""

        return

    def get_moments(self) -> List[np.ndarray]:
        return [self.mu, np.linalg.inv(self.Lambda) + np.outer(self.mu, self.mu)]


class Delta:
    """Moment calculator for a Dirac delta distribution

    Parameters
    ----------
    mu : np.ndarray
        Mean value

    """

    def __init__(self, mu: float):

        self.mu = mu
        """Mean vector"""

        return

    def get_moments(self) -> List[np.ndarray]:
        return [np.array(self.mu), np.array(self.mu) ** 2]


def create_gaussian_theta(formula: Formula):
    return Gaussian(*formula.prior)


class GAM:
    """Generalized additive model with NumPy backend

    Parameters
    ----------
    formula : gammy.formulae.Formula
        Formula object containing the terms and prior
    theta : Gaussian
        Model parameters vector
    tau : Delta
        Observation noise precision (inverse variance)

    """

    def __init__(self, formula, tau, theta=None) -> None:

        self.formula = formula
        """Model formula"""

        self.tau = tau
        """Additive noise precision parameter"""

        self.theta = (
            theta if theta is not None else create_gaussian_theta(formula)
        )
        """Model parameters"""

    def __len__(self) -> int:
        """Number of model parameters

        """
        return len(sum(self.formula.terms, []))

    @property
    def theta_marginals(self) -> List[Gaussian]:
        """Marginal distributions of model parameters

        """
        u = self.theta.get_moments()
        mus = utils.unflatten(u[0], self.formula.terms)
        covs = utils.extract_diag_blocks(
            utils.solve_covariance(u),
            self.formula.terms
        )
        return [
            Gaussian(mu=mu, Lambda=np.linalg.inv(cov))
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

    def theta_marginal(self, i: int) -> Gaussian:
        """Extract marginal distribution for a specific term

        """
        u = self.theta.get_moments()
        mus = utils.unflatten(u[0], self.formula.terms)
        covs = utils.extract_diag_blocks(
            utils.solve_covariance(u),
            self.formula.terms
        )
        return Gaussian(
            mu=mus[i],
            Lambda=np.linalg.inv(covs[i])
        )

    def fit(self, input_data, y) -> "GAM":
        """Estimate model parameters

        Parameters
        ----------
        input_data : np.ndarray
            Input data
        y : np.ndarray
            Observations

        """
        X = self.formula.design_matrix(input_data)
        #
        # NOTE: Posterior covariance formula based on Kaipio--Somersalo; Remark
        # after Theorem 3.7
        #
        # Perhaps we should use the other posterior covariance formulation which
        # doesn't require directly inverting any matrix. Then we could get rid
        # of all calls of `np.linalg.inv` as long as we refactor out references
        # to the precision matrices `Lambda`. The downside is we would need to
        # refactor also `gammy.models.bayespy.GAM` if want to keep similar
        # interfaces.
        #
        Lambda_post = self.theta.Lambda + self.tau.mu * np.dot(X.T, X)
        mu_post = np.linalg.solve(
            Lambda_post,
            np.dot(self.tau.mu * X.T, y) + np.dot(self.theta.Lambda, self.theta.mu)
        )
        return GAM(
            formula=self.formula,
            tau=self.tau,
            theta=Gaussian(mu=mu_post, Lambda=Lambda_post)
        )

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
        Sigma = utils.solve_covariance(self.theta.get_moments())
        return (
            np.dot(X, self.theta.mu),
            # Based on formula: var(A x) = A var(x) A'
            np.diag(np.dot(X, np.dot(Sigma, X.T))) + self.inv_mean_tau
        )

    def predict_variance_theta(self, input_data) -> Tuple[np.ndarray]:
        """Predict observations with variance from model parameters

        Parameters
        ----------
        input_data : np.ndarray

        """
        X = self.formula.design_matrix(input_data)
        Sigma = utils.solve_covariance(self.theta.get_moments())
        return (
            np.dot(X, self.theta.mu),
            np.diag(np.dot(X, np.dot(Sigma, X.T)))
        )

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

        """
        Xs = [
            self.formula.design_matrix(input_data, i)
            for i in range(len(self.formula))
        ]
        Sigmas = [
            utils.solve_covariance(theta.get_moments())
            for theta in self.theta_marginals
        ]
        mus = [np.dot(X, c) for (X, c) in zip(Xs, self.mean_theta)]
        sigmas = [
            np.diag(np.dot(X, np.dot(Sigma, X.T)))
            for (X, Sigma) in zip(Xs, Sigmas)
        ]
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
        """Evaluate mean and variance for a given term

        Parameters
        ----------
        input_data : np.ndarray

        """
        X = self.formula.design_matrix(input_data, i)
        Sigma = utils.solve_covariance(self.theta_marginal(i).get_moments())
        mu = np.dot(X, self.mean_theta[i])
        sigma = np.diag(np.dot(X, np.dot(Sigma, X.T)))
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
            Input data
        y : np.ndarray
            Observations

        """
        mus = self.predict_marginals(input_data)
        return y - np.sum(mus[:i] + mus[i + 1:], axis=0)

    def save(self, filepath: str) -> None:
        """Save the model to disk

        Supported file formats: JSON and HDF5

        """
        file_ext = filepath.split(".")[-1]
        if file_ext in ("h5", "hdf5"):
            with h5py.File(filepath, "w") as h5f:
                theta_group = h5f.create_group("theta")
                tau_group = h5f.create_group("tau")
                utils.write_to_hdf5(theta_group, self.theta.mu, "mu")
                utils.write_to_hdf5(theta_group, self.theta.Lambda, "Lambda")
                utils.write_to_hdf5(tau_group, self.tau.mu, "mu")
        elif file_ext == "json":
            with open(filepath, "w+") as jsonf:
                json.dump(
                    obj={
                        "theta": {
                            "mu": self.theta.mu.tolist(),
                            "Lambda": self.theta.Lambda.tolist()
                        },
                        "tau": {
                            "mu": self.tau.mu
                        }
                    },
                    fp=jsonf
                )
        else:
            raise ValueError(f"Unknown file type: {file_ext}")
        return

    def load(self, filepath: str) -> "GAM":
        """Load model from a file on disk

        """
        file_ext = filepath.split(".")[-1]
        if file_ext in ("h5", "hdf5"):
            with h5py.File(filepath, "r") as h5f:
                tau = Delta(mu=h5f["tau"]["mu"][...])
                theta = Gaussian(
                    mu=h5f["theta"]["mu"][...],
                    Lambda=h5f["theta"]["Lambda"][...]
                )
        elif file_ext == "json":
            with open(filepath, "r") as jsonf:
                raw = json.load(jsonf)
                tau = Delta(raw["tau"]["mu"])
                theta = Gaussian(
                    mu=raw["theta"]["mu"],
                    Lambda=raw["theta"]["Lambda"]
                )
        else:
            raise ValueError(f"Unknown file type: {file_ext}")
        return GAM(formula=self.formula, theta=theta, tau=tau)


def LinearModel(n_features, prior=None, **kwargs):
    prior = (
        (np.zeros(n_features), 1e-6 * np.identity(n_features)) if prior is None
        else prior
    )
    basis = [(lambda t: t) for i in range(n_features)]
    formula = Formula(terms=[basis], prior=prior)
    return GAM(formula, **kwargs)
