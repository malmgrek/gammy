import logging

import bayespy as bp
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    logging.info(
        "Problem with importing Axes3D from mpl_toolkits.mplot3d. Skipping."
    )
import matplotlib.pyplot as plt
import numpy as np

from gammy import utils
from gammy.utils import pipe


def validation_plot(
        model,
        input_data,
        y,
        grid_limits,
        input_maps,
        index=None,
        xlabels=None,
        titles=None,
        ylabel=None,
        gridsize=20,
        color="r",
        **kwargs
):
    """Generic validation plot for a GAM

    """
    # TODO: Support larger input dimensions
    index = np.arange(len(input_data)) if index is None else index

    # Figure definitions
    N = len(model.formula)
    fig = plt.figure(figsize=(8, max(4 * N // 2, 8)))
    gs = fig.add_gridspec(N // 2 + 3, 2)
    xlabels = xlabels or [None] * len(model.formula)
    titles = titles or [None] * len(model.formula)

    # Data and predictions
    grid = (
        pipe(
            grid_limits,
            utils.listmap(lambda x: np.linspace(x[0], x[1], gridsize)),
            lambda x: np.array(x).T
        ) if len(input_data.shape) > 1
        else np.linspace(grid_limits[0], grid_limits[1], gridsize)
    )
    marginals = model.predict_variance_marginals(grid)
    residuals = model.marginal_residuals(input_data, y)

    # Time series plot
    ax = fig.add_subplot(gs[0, :])
    (mu, sigma_theta) = model.predict_variance_theta(input_data)
    lower = mu - 2 * np.sqrt(sigma_theta + model.inv_mean_tau)
    upper = mu + 2 * np.sqrt(sigma_theta + model.inv_mean_tau)
    ax.plot(index, y, linewidth=0, marker="o", alpha=0.3, color=color)
    ax.plot(index, mu, color="k")
    ax.fill_between(index, lower, upper, color="k", alpha=0.3)
    ax.grid(True)

    # XY-plot
    ax = fig.add_subplot(gs[1, :])
    ax.plot(mu, y, alpha=0.3, marker="o", lw=0, color=color)
    ax.plot([mu.min(), mu.max()], [mu.min(), mu.max()], c="k", label="x=y")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Observations")

    # Partial residual plots
    for i, ((mu, sigma), res, input_map, xlabel, title) in enumerate(
        zip(marginals, residuals, input_maps, xlabels, titles)
    ):
        x = input_map(grid)
        if len(x.shape) == 1 or x.shape[1] == 1:
            ax = fig.add_subplot(gs[2 + i // 2, i % 2])
            (lower, upper) = (
                mu - 2 * np.sqrt(sigma),
                mu + 2 * np.sqrt(sigma)
            )
            ax.scatter(input_map(input_data), res, color=color, **kwargs)
            ax.plot(x, mu, c='k', lw=2)
            ax.fill_between(x, lower, upper, alpha=0.3, color="k")
            ax.set_xlabel(xlabel)
        elif x.shape[1] == 2:
            ax = fig.add_subplot(gs[2 + i // 2, i % 2], projection="3d")
            u, v = np.meshgrid(x[:, 0], x[:, 1])
            w = np.hstack((
                u.reshape(-1, 1), v.reshape(-1, 1)
            ))
            # Override mu and sigma on purpose!
            (mu, sigma) = model.predict_variance_marginal(w, i)
            mu_mesh = mu.reshape(u.shape)
            ax.plot_surface(u, v, mu_mesh)
        else:
            raise NotImplementedError("High-dimensional plots not supported.")
        ax.set_title(title)
        ax.grid(True)

    fig.tight_layout()
    return fig


def gaussian1d_density_plot(model, grid_limits=[0.5, 1.5]):
    """Plot 1-D density for each parameter

    Parameters
    ----------
    grid_limits : list
        Grid of `tau` has endpoints `[grid_limits[0] * mu, grid_limits[1] * mu]`
        where `mu` is the expectation of `tau`.

    """
    N = len(model.formula)
    fig = plt.figure(figsize=(8, max(4 * N // 2, 8)))
    gs = fig.add_gridspec(N + 1, 1)

    # Plot inverse gamma
    ax = fig.add_subplot(gs[0])
    (b, a) = (-model.tau.phi[0], model.tau.phi[1])
    mu = a / b
    grid = np.arange(0.5 * mu, 1.5 * mu, mu / 300)
    ax.plot(grid, model.tau.pdf(grid))
    ax.set_title(r"$\tau$ = noise inverse variance")
    ax.grid(True)

    # Plot marginal thetas
    for i, theta in enumerate(model.theta_marginals):
        ax = fig.add_subplot(gs[i + 1])
        mus = theta.get_moments()[0]
        mus = np.array([mus]) if mus.shape == () else mus
        cov = utils.solve_covariance(theta)
        stds = pipe(
            np.array([cov]) if cov.shape == ()
            else np.diag(cov),
            np.sqrt
        )
        left = (mus - 4 * stds).min()
        right = (mus + 4 * stds).max()
        grid = np.arange(left, right, (right - left) / 300)
        for (mu, std) in zip(mus, stds):
            node = bp.nodes.GaussianARD(mu, 1 / std ** 2)
            ax.plot(grid, node.pdf(grid))
        ax.set_title(r"$\theta_{0}$".format(i))
        ax.grid(True)

    fig.tight_layout()
    return fig


def gaussian2d_density_plot(model, i, j):
    """Plot 2-D joint distribution of indices i and j

    """
    raise NotImplementedError


def covariance_plot(model):
    """Covariance matrix

    """
    raise NotImplementedError


def basis_plot(model, grid_limits, input_maps, gridsize=20):
    """Plot all basis functions

    """
    # Figure definition
    N = len(model.formula)
    fig = plt.figure(figsize=(8, max(4 * N // 2, 8)))
    gs = fig.add_gridspec(N, 1)

    # Data and predictions
    grid = (
        pipe(
            grid_limits,
            utils.listmap(lambda x: np.linspace(x[0], x[1], gridsize)),
            lambda x: np.array(x).T
        )
    )

    # Plot stuff
    for i, (basis, input_map) in enumerate(
            zip(model.formula.bases, input_maps)
        ):
        ax = fig.add_subplot(gs[i])
        x = input_map(grid)
        for f in basis:
            ax.plot(x, f(grid))

    return fig
