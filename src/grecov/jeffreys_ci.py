"""Bayesian credible interval for the multinomial weighted mean.

Uses the Dirichlet-multinomial conjugate posterior with a Jeffreys
(half-unit) prior: posterior ∝ Dirichlet(counts + 1/2).  The interval
is computed by numerically inverting the CDF of the weighted sum
Y = v^T P where P ~ Dirichlet(alpha_post).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

# ── Fixed quadrature for the Gil-Pelaez integral ─────────────────────────────
# Gauss-Legendre on [0,1] with substitution u = t/(1-t) maps [0,∞).
# 64 nodes gives ~15 digits of accuracy for this integrand.

_N_QUAD = 64
_leg_nodes, _leg_weights = np.polynomial.legendre.leggauss(_N_QUAD)
_t = 0.5 * (_leg_nodes + 1)  # map [-1,1] → [0,1]
_u = _t / (1 - _t)
_quad_weights = 0.5 * _leg_weights / (1 - _t) ** 2  # includes Jacobian
_inv_u = 1.0 / _u


def _dirichlet_weighted_sum_cdf(alpha, weights, y):
    """CDF of Y = sum_i weights[i] * P_i, P ~ Dirichlet(alpha).

    Uses the Gil-Pelaez inversion formula with a fixed Gauss-Legendre
    quadrature (vectorized over all nodes at once).
    """
    lo, hi = float(np.min(weights)), float(np.max(weights))

    if y <= lo:
        return 0.0
    if y >= hi:
        return 1.0

    lam = weights - y
    lu = np.outer(_u, lam)  # (n_nodes, k)
    theta = np.arctan(lu) @ alpha
    log_denom = 0.5 * (np.log1p(lu * lu) @ alpha)
    vals = np.sin(theta) * np.exp(-log_denom) * _inv_u
    integral = _quad_weights @ vals
    return max(0.0, min(1.0, 0.5 - integral / math.pi))


def _dirichlet_weighted_sum_ppf(alpha, weights, q, *, xtol=1e-6):
    """Quantile function (inverse CDF) of Y = sum_i weights[i] * P_i."""
    lo, hi = float(np.min(weights)), float(np.max(weights))
    if q <= 0.0:
        return lo
    if q >= 1.0:
        return hi

    return float(
        brentq(
            lambda y: _dirichlet_weighted_sum_cdf(alpha, weights, y) - q,
            lo,
            hi,
            xtol=xtol,
        )
    )


def jeffreys_ci(counts, values, alpha=0.05, *, tol=1e-6):
    """Bayesian credible interval for the multinomial weighted mean.

    Uses a Jeffreys prior (Dirichlet(1/2, ..., 1/2)) conjugate update.

    Parameters
    ----------
    counts : array-like of int
        Observed category counts.
    values : array-like of float
        Numerical value assigned to each category.
    alpha : float
        Significance level (default 0.05 for a 95% interval).
    tol : float
        Tolerance for root-finding.

    Returns
    -------
    dict with keys: lower, upper.
    """
    counts = np.asarray(counts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if len(counts) != len(values):
        raise ValueError("counts and values must have the same length")

    # Jeffreys posterior: Dirichlet(counts + 1/2)
    alpha_post = counts + 0.5

    lower = _dirichlet_weighted_sum_ppf(alpha_post, values, alpha / 2, xtol=tol)
    upper = _dirichlet_weighted_sum_ppf(alpha_post, values, 1 - alpha / 2, xtol=tol)

    return {"lower": lower, "upper": upper}
