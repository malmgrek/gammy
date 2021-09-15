"""GAM with "raw" NumPy backend

TODO:
- Match with the BayesPy interface `get_moments`

"""

import json
from typing import List, Tuple

import h5py
import numpy as np

import gammy
from gammy import utils
from gammy.formulae import Formula
from gammy.utils import listmap, pipe


class Gaussian:
    """Moment calculator a Gaussian distribution

    Parameters
    ----------
    mu : np.ndarray
        Mean value
    Lambda : np.ndarray
        Precision matrix (inverse of covariance)

    """

    def __init__(self, mu: np.ndarray, Lambda: np.ndarray) -> None:
        self.mu = mu
        self.Lambda = Lambda
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
        return

    def get_moments(self) -> List[np.ndarray]:
        return [np.array(self.mu), np.array(self.mu) ** 2]


def create_gaussian_theta(formula: Formula):
    return Gaussian(*formula.prior)


# TODO: Rename to GAM
# TODO: Add docstrings to methods
# TODO: Uniformize docstring and type hint styles with bayespy.GAM
class LinearGAM:

    def __init__(self, formula: Formula, tau: Delta, theta: Gaussian=None) -> None:
        self.formula = formula
        self.tau = tau
        self.theta = (
            theta if theta is not None else create_gaussian_theta(formula)
        )

    def __len__(self) -> int:
        """Number of model parameters

        """
        return len(utils.flatten(self.formula.bases))

    @property
    def theta_marginals(self) -> List:
        mus = utils.unflatten(self.theta.get_moments()[0], self.formula.bases)
        covs = utils.extract_diag_blocks(
            utils.solve_covariance(self.theta),
            self.formula.bases
        )
        return [
            Gaussian(mu=mu, Lambda=np.linalg.inv(cov))
            for (mu, cov) in zip(mus, covs)
        ]

    @property
    def mean_theta(self) -> List[np.ndarray]:
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
        mus = utils.unflatten(self.theta.get_moments()[0], self.formula.bases)
        covs = utils.extract_diag_blocks(
            utils.solve_covariance(self.theta),
            self.formula.bases
        )
        return Gaussian(
            mu=mus[i],
            Lambda=np.linalg.inv(covs[i])
        )

    def fit(self, input_data: np.ndarray, y: np.ndarray):
        X = self.formula.build_X(input_data)
        # Kaipio--Somersalo; Remark after Theorem 3.7
        Lambda_post = self.theta.Lambda + self.tau.mu * np.dot(X.T, X)
        mu_post = np.linalg.solve(
            Lambda_post,
            np.dot(self.tau.mu * X.T, y) + np.dot(self.theta.Lambda, self.theta.mu)
        )
        return LinearGAM(
            formula=self.formula,
            tau=self.tau,
            theta=Gaussian(mu=mu_post, Lambda=Lambda_post)
        )

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        X = self.formula.build_X(input_data)
        return np.dot(X, np.hstack(self.mean_theta))

    def predict_variance(
            self,
            input_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = self.formula.build_X(input_data)
        Sigma = utils.solve_covariance(self.theta)
        return (
            np.dot(X, self.theta.mu),
            # Based on formula: var(A x) = A var(x) A'
            np.diag(np.dot(X, np.dot(Sigma, X.T))) + self.inv_mean_tau
        )

    def predict_variance_theta(
            self,
            input_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = self.formula.build_X(input_data)
        Sigma = utils.solve_covariance(self.theta)
        return (
            np.dot(X, self.theta.mu),
            np.diag(np.dot(X, np.dot(Sigma, X.T)))
        )

    def predict_marginals(self, input_data: np.ndarray) -> List[np.ndarray]:
        Xs = self.formula.build_Xs(input_data)
        return [np.dot(X, c) for (X, c) in zip(Xs, self.mean_theta)]

    def predict_variance_marginals(
            self,
            input_data: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        Xs = self.formula.build_Xs(input_data)
        Sigmas = utils.listmap(utils.solve_covariance)(self.theta_marginals)
        mus = [np.dot(X, c) for (X, c) in zip(Xs, self.mean_theta)]
        sigmas = [
            np.diag(np.dot(X, np.dot(Sigma, X.T)))
            for (X, Sigma) in zip(Xs, Sigmas)
        ]
        return list(zip(mus, sigmas))

    def predict_marginal(self, input_data: np.ndarray, i: int) -> np.ndarray:
        X = self.formula.build_Xi(input_data, i=i)
        return np.dot(X, self.mean_theta[i])

    def predict_variance_marginal(
            self,
            input_data: np.ndarray,
            i: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = self.formula.build_Xi(input_data, i=i)
        Sigma = utils.solve_covariance(self.theta_marginal(i))
        mu = np.dot(X, self.mean_theta[i])
        sigma = np.diag(np.dot(X, np.dot(Sigma, X.T)))
        return (mu, sigma)

    def marginal_residuals(
            self,
            input_data: np.ndarray,
            y: np.ndarray
    ) -> List[np.ndarray]:
        mus = self.predict_marginals(input_data)
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
        mus = self.predict_marginals(input_data)
        return y - np.sum(mus[:i] + mus[i + 1:], axis=0)

    def save(self, filepath: str) -> None:
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

    def load(self, filepath: str):
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
        return LinearGAM(formula=self.formula, theta=theta, tau=tau)
