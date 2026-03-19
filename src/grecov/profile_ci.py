"""Profile likelihood confidence interval for the multinomial weighted mean.

Inverts the profile likelihood ratio test statistic for mu = v^T p,
using the chi-squared(1) critical value. The profile LR is computed
in closed form using the dual (Lagrangian) formulation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2


def _lambda_root(phat, a, tol=1e-12):
    """Solve for the Lagrange multiplier in the interior case.

    Finds lambda such that sum_i phat_i * a_i / (1 + lambda * a_i) = 0,
    where a_i = x_i - mu and phat_i = c_i / n (restricted to positive counts).
    """
    g0 = np.sum(phat * a)
    if abs(g0) < tol:
        return 0.0

    pos = a > 0
    neg = a < 0

    lower = float(np.max(-1.0 / a[pos])) if np.any(pos) else -np.inf
    upper = float(np.min(-1.0 / a[neg])) if np.any(neg) else np.inf

    # Move strictly inside the feasible interval
    lower = np.nextafter(lower, np.inf) if np.isfinite(lower) else -1e12
    upper = np.nextafter(upper, -np.inf) if np.isfinite(upper) else 1e12

    def g(lam):
        den = 1.0 + lam * a
        if np.any(den <= 0):
            j = np.where(den <= 0)[0][0]
            return np.sign(a[j]) * np.inf
        return np.sum(phat * a / den)

    return brentq(g, lower, upper)


def _profile_lr(counts, values, mu):
    """Profile likelihood ratio statistic -2 log R(mu).

    Parameters
    ----------
    counts : np.ndarray
        Observed counts (may contain zeros).
    values : np.ndarray
        Category values.
    mu : float
        Hypothesized mean.

    Returns
    -------
    float
        The profile LR statistic (non-negative; inf if mu is outside support).
    """
    xmin, xmax = float(values.min()), float(values.max())
    if not (xmin < mu < xmax):
        return np.inf

    mask = counts > 0
    c = counts[mask]
    xs = values[mask]
    n = c.sum()
    phat = c / n
    a = xs - mu

    lo, hi = float(xs.min()), float(xs.max())

    if lo <= mu <= hi:
        # Interior: solve 1D dual
        lam = _lambda_root(phat, a)
        return float(2.0 * np.sum(c * np.log1p(lam * a)))

    if mu < lo:
        # Lower tail: only xmin boundary is active
        return float(2.0 * np.sum(c * np.log((xs - xmin) / (mu - xmin))))

    # Upper tail: only xmax boundary is active
    return float(2.0 * np.sum(c * np.log((xmax - xs) / (xmax - mu))))


def profile_ci(counts, values, alpha=0.05):
    """Profile likelihood confidence interval for the multinomial weighted mean.

    Inverts the likelihood ratio test using a chi-squared(1) critical value.

    Parameters
    ----------
    counts : array-like of int
        Observed category counts.
    values : array-like of float
        Numerical value assigned to each category.
    alpha : float
        Significance level (default 0.05 for a 95% CI).

    Returns
    -------
    dict with keys: lower, upper.
    """
    counts = np.asarray(counts, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(counts) != len(values):
        raise ValueError("counts and values must have the same length")
    n = counts.sum()
    if n <= 0:
        raise ValueError("counts must sum to a positive number")

    muhat = float(np.dot(values, counts) / n)
    crit = float(chi2.ppf(1.0 - alpha, df=1))

    xmin, xmax = float(values.min()), float(values.max())
    eps = 1e-10

    def h(mu):
        return _profile_lr(counts, values, mu) - crit

    lower = float(brentq(h, xmin + eps, muhat - eps))
    upper = float(brentq(h, muhat + eps, xmax - eps))

    return {"lower": lower, "upper": upper}
