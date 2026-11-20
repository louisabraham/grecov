"""Gaffke confidence interval for the multinomial weighted mean.

For X ~ Mult(n, p) and mu = v^T p, the interval is built from two
Dirichlet posteriors that each add one pseudo-count at an extreme
category:

    P_- ~ Dir(x + e_min),   P_+ ~ Dir(x + e_max)
    CI  = [ Q_{alpha/2}(v^T P_-),  Q_{1-alpha/2}(v^T P_+) ]

where e_min / e_max are the indicators of the categories carrying the
smallest / largest value.  There is no optimisation over the simplex
and no enumeration of multinomial outcomes: each endpoint is one
scalar quantile of a Dirichlet-weighted sum.

Categories with a zero Dirichlet parameter contribute no mass and are
dropped, which is the correct limit of Dir(alpha) as alpha_i -> 0.
"""

from __future__ import annotations

import numpy as np

from grecov.jeffreys_ci import _dirichlet_weighted_sum_ppf


def _quantile(alpha_post, values, q, tol):
    """Quantile of v^T P, P ~ Dir(alpha_post), dropping null coordinates."""
    keep = alpha_post > 0
    a = alpha_post[keep]
    w = values[keep]
    if len(a) == 1:
        return float(w[0])
    return _dirichlet_weighted_sum_ppf(a, w, q, xtol=tol)


def gaffke_ci(counts, values, alpha=0.05, *, tol=1e-6):
    """Gaffke confidence interval for the multinomial weighted mean.

    Parameters
    ----------
    counts : array-like of int
        Observed category counts.
    values : array-like of float
        Numerical value assigned to each category.
    alpha : float
        Significance level (default 0.05 for a 95% CI).
    tol : float
        Tolerance for the scalar root finder.

    Returns
    -------
    dict with keys: lower, upper.
    """
    counts = np.asarray(counts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if len(counts) != len(values):
        raise ValueError("counts and values must have the same length")

    i_min = int(np.argmin(values))
    i_max = int(np.argmax(values))

    a_lo = counts.copy()
    a_lo[i_min] += 1.0
    a_hi = counts.copy()
    a_hi[i_max] += 1.0

    lower = _quantile(a_lo, values, alpha / 2.0, tol)
    upper = _quantile(a_hi, values, 1.0 - alpha / 2.0, tol)

    return {"lower": float(lower), "upper": float(upper)}
