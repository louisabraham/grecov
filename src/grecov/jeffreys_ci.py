"""Bayesian credible interval for the multinomial weighted mean.

Uses the Dirichlet-multinomial conjugate posterior with a Jeffreys
(half-unit) prior: posterior ∝ Dirichlet(counts + 1/2).  The interval
is computed by numerically inverting the CDF of the weighted sum
Y = v^T P where P ~ Dirichlet(alpha_post).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def _dirichlet_weighted_sum_cdf(
    alpha, weights, y, *, epsabs=1e-10, epsrel=1e-10, limit=400
):
    """CDF of Y = sum_i weights[i] * P_i, P ~ Dirichlet(alpha).

    Uses the inversion formula based on the characteristic function,
    reduced to a single real integral (Gil-Pelaez style).
    """
    lo, hi = float(weights[0]), float(weights[-1])

    if y <= lo:
        return 0.0
    if y >= hi:
        return 1.0

    lam = weights - y

    def integrand(u):
        if u == 0.0:
            return float(np.dot(alpha, lam))
        theta = np.sum(alpha * np.arctan(lam * u))
        log_denom = 0.5 * np.sum(alpha * np.log1p((lam * u) ** 2))
        return np.sin(theta) * np.exp(-log_denom) / u

    val, _ = quad(integrand, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return float(np.clip(0.5 - val / np.pi, 0.0, 1.0))


def _dirichlet_weighted_sum_ppf(
    alpha, weights, q, *, epsabs=1e-10, epsrel=1e-10, limit=400
):
    """Quantile function (inverse CDF) of Y = sum_i weights[i] * P_i."""
    lo, hi = float(weights[0]), float(weights[-1])
    if q <= 0.0:
        return lo
    if q >= 1.0:
        return hi

    def f(y):
        return (
            _dirichlet_weighted_sum_cdf(
                alpha, weights, y, epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            - q
        )

    return float(brentq(f, lo, hi, xtol=epsabs, rtol=epsrel))


def jeffreys_ci(counts, values, alpha=0.05):
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

    Returns
    -------
    dict with keys: lower, upper.
    """
    counts = np.asarray(counts, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(counts) != len(values):
        raise ValueError("counts and values must have the same length")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    # Jeffreys posterior: Dirichlet(counts + 1/2)
    alpha_post = counts + 0.5

    # Sort by value and merge equal-valued categories
    order = np.argsort(values)
    alpha_post = alpha_post[order]
    values_sorted = values[order]

    merged_v, merged_a = [], []
    for v, a in zip(values_sorted, alpha_post):
        if merged_v and np.isclose(v, merged_v[-1], atol=1e-15, rtol=0.0):
            merged_a[-1] += a
        else:
            merged_v.append(float(v))
            merged_a.append(float(a))

    weights = np.asarray(merged_v)
    alpha_merged = np.asarray(merged_a)

    if len(weights) == 1:
        return {"lower": weights[0], "upper": weights[0]}

    lower = _dirichlet_weighted_sum_ppf(alpha_merged, weights, alpha / 2)
    upper = _dirichlet_weighted_sum_ppf(alpha_merged, weights, 1 - alpha / 2)

    return {"lower": lower, "upper": upper}
