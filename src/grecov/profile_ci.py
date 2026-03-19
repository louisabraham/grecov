"""Profile likelihood confidence interval for the multinomial weighted mean.

Inverts the profile likelihood ratio test statistic for mu = v^T p,
using the chi-squared(1) critical value. The profile LR is computed
in closed form using the dual (Lagrangian) formulation.

Uses pure-Python arithmetic to avoid numpy overhead on small arrays.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import chi2


def _profile_lr(c_pos, x_pos, phat, xmin, xmax, mu):
    """Profile likelihood ratio statistic -2 log R(mu).

    All inputs are plain Python lists (positive-count categories only).
    """
    if not (xmin < mu < xmax):
        return float("inf")

    a = [xi - mu for xi in x_pos]
    lo_x = min(x_pos)
    hi_x = max(x_pos)

    if lo_x <= mu <= hi_x:
        # Interior: solve for Lagrange multiplier
        g0 = sum(pi * ai for pi, ai in zip(phat, a))
        if abs(g0) < 1e-12:
            return 0.0

        lower = -1e12
        upper = 1e12
        for ai in a:
            if ai > 0:
                lower = max(lower, -1.0 / ai)
            elif ai < 0:
                upper = min(upper, -1.0 / ai)
        lower = math.nextafter(lower, math.inf)
        upper = math.nextafter(upper, -math.inf)

        lam = brentq(
            lambda lam: sum(pi * ai / (1.0 + lam * ai) for pi, ai in zip(phat, a)),
            lower,
            upper,
        )
        return 2.0 * sum(ci * math.log1p(lam * ai) for ci, ai in zip(c_pos, a))

    if mu < lo_x:
        return 2.0 * sum(
            ci * math.log((xi - xmin) / (mu - xmin)) for ci, xi in zip(c_pos, x_pos)
        )

    return 2.0 * sum(
        ci * math.log((xmax - xi) / (xmax - mu)) for ci, xi in zip(c_pos, x_pos)
    )


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
    counts = [float(c) for c in counts]
    values = [float(v) for v in values]

    if len(counts) != len(values):
        raise ValueError("counts and values must have the same length")

    n = sum(counts)
    if n <= 0:
        raise ValueError("counts must sum to a positive number")

    # Precompute: only positive-count categories
    c_pos = []
    x_pos = []
    for ci, xi in zip(counts, values):
        if ci > 0:
            c_pos.append(ci)
            x_pos.append(xi)
    phat = [ci / n for ci in c_pos]

    muhat = sum(v * c for v, c in zip(values, counts)) / n
    crit = float(chi2.ppf(1.0 - alpha, df=1))
    xmin, xmax = min(values), max(values)
    eps = 1e-10

    def h(mu):
        return _profile_lr(c_pos, x_pos, phat, xmin, xmax, mu) - crit

    lower = float(brentq(h, xmin + eps, muhat - eps))
    upper = float(brentq(h, muhat + eps, xmax - eps))

    return {"lower": lower, "upper": upper}
