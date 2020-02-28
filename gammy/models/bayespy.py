import bayespy as bp
import h5py
import numpy as np

from gammy import utils
from gammy.utils import listmap, pipe


def build_theta(formula):
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
        Observation noise

    Currently tau is fixed to Gamma distribution, i.e., it is not possible
    to manually fix the noise level. Note though that one can set tight values
    for `α, β` in `Gamma(α, β)`, recalling that `mean = α / β` and
    `variance = α / β ** 2`. The upside is that by estimating the noise level,
    one gets a nice prediction uncertainty estimate.

    """
    # TODO: Statistics in original coordinates?

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
        self.theta = theta if theta is not None else build_theta(formula)

    def __len__(self):
        return len(self.formula.bases)

    @property
    def theta_marginals(self) -> list:
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
    def mean_theta(self) -> list:
        return pipe(
            self.theta.get_moments()[0],
            lambda x: utils.unflatten(x, self.formula.bases),
            listmap(np.array)
        )

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
        self, input_data: np.ndarray, y: np.ndarray, repeat: int=1000, **kwargs
    ):
        """Update BayesPy nodes and construct a GAM predictor

        WARNING: Currently mutates the original object's ``theta`` and ``tau``.

        An option to "reset" the original object to prior is to use the method
        ``initialize_from_prior()`` of BayesPy nodes.

        """
        # TODO: Test that fit always gives same result (if theta reset)
        X = self.formula.build_X(input_data)
        F = bp.nodes.SumMultiply("i,i", self.theta, X)
        Y = bp.nodes.GaussianARD(F, self.tau)
        Y.observe(y)
        Q = bp.inference.VB(Y, self.theta, self.tau)
        Q.update(repeat=repeat, **kwargs)
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

    def predict_variance(self, input_data: np.ndarray) -> tuple:
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

    def predict_variance_theta(self, input_data: np.ndarray) -> tuple:
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

    def predict_marginals(self, input_data: np.ndarray) -> list:
        """Predict all terms separately

        """
        Xs = self.formula.build_Xs(input_data)
        return [np.dot(X, c) for X, c in zip(Xs, self.mean_theta)]

    def predict_variance_marginals(self, input_data: np.ndarray) -> list:
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
        self, input_data: np.ndarray, i: int
    ) -> tuple:
        # Not refactored with predict_marginal for perf reasons
        X = self.formula.build_Xi(input_data, i=i)
        F = bp.nodes.SumMultiply("i,i", self.theta_marginal(i), X)
        mu = np.dot(X, self.mean_theta[i])
        sigma = pipe(F, utils.solve_covariance, np.diag)
        return (mu, sigma)

    def marginal_residuals(
        self, input_data: np.ndarray, y: np.ndarray
    ) -> list:
        """Marginal (partial) residuals

        """
        marginals = self.predict_variance_marginals(input_data)
        mus = [mu for (mu, _) in marginals]
        return [
            y - np.sum(mus[:i] + mus[i + 1:], axis=0)
            for i in range(len(mus))
        ]

    def marginal_residual(
        self, input_data: np.ndarray, y: np.ndarray, i: int
    ) -> np.ndarray:
        # Not refactored with marginal_residuals for perf reasons
        marginals = self.predict_variance_marginals(input_data)
        mus = [mu for (mu, _) in marginals]
        return y - np.sum(mus[:i] + mus[i + 1:], axis=0)

    def _save(self, group):
        self.tau._save(group.create_group("tau"))
        self.theta._save(group.create_group("theta"))
        return group

    def save(self, filename) -> None:
        # TODO: OS independent filepaths
        with h5py.File(filename, "w") as h5f:
            group = h5f.create_group("nodes")
            self._save(group)
        return

    def _load(self, h5f):
        tau = self.tau
        tau._load(h5f["nodes"]["tau"])
        theta = build_theta(self.formula)
        theta._load(h5f["nodes"]["theta"])
        return BayesianGAM(
            formula=self.formula,
            tau=tau,
            theta=theta
        )

    def load(self, filename):
        # TODO: OS independent filepaths
        with h5py.File(filename, "r") as h5f:
            return self._load(h5f)
