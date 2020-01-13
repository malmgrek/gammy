import attr
import bayespy as bp
import h5py
import numpy as np

from gammy import utils
from gammy.utils import listmap, pipe


@attr.s(frozen=True)
class BayesianGAM():
    """Generalized additive model predictor

    Parameters
    ----------
    formula : BayesPyFormula
    theta : bp.nodes.Gaussian
    tau : bp.nodes.Gamma

    Currently tau is fixed to Gamma distribution, i.e., it is not possible
    to manually fix the noise level. Note though that one can set tight values
    for `α, β` in `Gamma(α, β)`, recalling that `mean = α / β` and
    `variance = α / β ** 2`.

    """
    # TODO: Parameter uncertainty
    # TODO: Prediction uncertainty
    # TODO: Statistics in original coordinates

    formula = attr.ib()
    tau = attr.ib(factory=lambda: bp.nodes.Gamma(1e-3, 1e-3))
    theta = attr.ib(default=None)

    def __len__(self):
        return len(self.formula.bases)

    @property
    def thetas(self):
        """Nodes for the basis specific weight collections

        """
        # TODO: Test that the partial distributions are correct
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
    def mean_theta(self):
        return pipe(
            self.theta.get_moments()[0],
            lambda x: utils.unflatten(x, self.formula.bases),
            listmap(lambda x: np.array(x))
        )

    @property
    def mean_inv_sqrt_tau(self):
        # "Unexplained noise" covariance estimate
        # Should work also if tau is an 1-D precision "matrix"
        Y = bp.nodes.GaussianARD(np.array([0]), self.tau)
        return np.sqrt(Y.get_moments()[1])[0]

    def extract_thetai(self, i):
        mus = utils.unflatten(self.theta.get_moments()[0], self.formula.bases)
        covs = utils.extract_diag_blocks(
            utils.solve_covariance(self.theta),
            self.formula.bases
        )
        return bp.nodes.Gaussian(
            mu=mus[i],
            Lambda=bp.utils.linalg.inv(covs[i])
        )

    def fit(self, input_data, y, repeat=1000, **kwargs):
        """Update BayesPy nodes and construct a GAM predictor

        """
        (theta, _, tau, _) = utils.update(
            formula=self.formula,
            input_data=input_data,
            y=y,
            tau=self.tau,
            repeat=repeat,
            **kwargs
        )
        return BayesianGAM(
            formula=self.formula,
            tau=tau,
            theta=theta
        )

    def predict(self, input_data):
        X = self.formula.build_X(input_data)
        return np.dot(X, np.hstack(self.mean_theta))

    def predict_theta_uncertainty(self, input_data, scale=2.0):
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
            scale * pipe(F, utils.solve_covariance, np.diag, np.sqrt)
        )

    def predict_total_uncertainty(self, input_data, scale=2.0):
        X = self.formula.build_X(input_data)
        F = bp.nodes.SumMultiply("i,i", self.theta, X)
        return (
            F.get_moments()[0],
            (
                scale * pipe(F, utils.solve_covariance, np.diag, np.sqrt) +
                scale * self.mean_inv_sqrt_tau
            )
        )

    def predict_partials(self, input_data, scale=2.0):
        Xs = self.formula.build_Xs(input_data)
        Fs = [
            bp.nodes.SumMultiply("i,i", theta, X)
            for X, theta in zip(Xs, self.thetas)
        ]
        mus = [np.dot(X, c) for X, c in zip(Xs, self.mean_theta)]
        margins = [
            scale * pipe(F, utils.solve_covariance, np.diag, np.sqrt)
            for F in Fs
        ]
        return list(zip(mus, margins))

    def predict_partial(self, input_data, i, scale=2.0):
        # Not refactored with predict_partials for perf reasons
        X = self.formula.build_Xi(input_data, i=i)
        F = bp.nodes.SumMultiply("i,i", self.extract_thetai(i), X)
        mu = np.dot(X, self.mean_theta[i])
        margin = scale * pipe(F, utils.solve_covariance, np.diag, np.sqrt)
        return (mu, margin)


    def partial_residuals(self, input_data, y):
        partials = self.predict_partials(input_data)
        mus = [mu for (mu, _) in partials]
        return [
            y - np.sum(mus[:i] + mus[i + 1:], axis=0)
            for i in range(len(mus))
        ]

    def partial_residual(self, input_data, y, i):
        # Not refactored with partial_residuals for perf reasons
        partials = self.predict_partials(input_data)
        mus = [mu for (mu, _) in partials]
        return y - np.sum(mus[:i] + mus[i + 1:], axis=0)

    def save(self, filename):
        # TODO: OS independent filepaths
        with h5py.File(filename, "w") as h5f:
            group = h5f.create_group("nodes")
            self.tau._save(group.create_group("tau"))
            self.theta._save(group.create_group("theta"))
        return

    def load(self, filename):
        # TODO: OS independent filepaths
        with h5py.File(filename, "r") as h5f:
            tau = self.tau
            tau._load(h5f["nodes"]["tau"])
            theta = self.formula.build_theta()
            theta._load(h5f["nodes"]["theta"])
        return BayesianGAM(
            formula=self.formula,
            tau=tau,
            theta=theta
        )