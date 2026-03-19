"""Profile likelihood confidence interval for the multinomial weighted mean.

Uses Owen's endpoint parameterization with Newton's method.

Instead of a nested root-find (outer in mu, inner in the Lagrange
multiplier lambda), we parameterize directly by lambda and solve
a single equation LR(lambda) = chi2_crit per endpoint.

Given lambda (outside the range of observed values):
  - p_k(lambda) proportional to n_k / (x_k - lambda)
  - mu(lambda) = lambda + n / sum(n_k / (x_k - lambda))
  - LR(lambda) = 2 * sum n_k * log((x_k - lambda) / (n * A))
"""

from __future__ import annotations

import math

from scipy.stats import chi2


def profile_ci(counts, values, alpha=0.05):
    """Profile likelihood confidence interval for the multinomial weighted mean.

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
    n = sum(float(c) for c in counts)
    if n <= 0:
        raise ValueError("counts must sum to a positive number")

    crit = float(chi2.ppf(1.0 - alpha, df=1))
    xmin = min(float(v) for v in values)
    xmax = max(float(v) for v in values)

    # Keep only categories with positive counts
    obs = [(float(c), float(v)) for c, v in zip(counts, values) if c > 0]
    x_lo = min(x for _, x in obs)
    x_hi = max(x for _, x in obs)

    # Degenerate: all mass on one value
    if x_lo == x_hi:
        scale = math.exp(-crit / (2 * n))
        lo = xmin if x_lo == xmin else xmin + (x_lo - xmin) * scale
        hi = xmax if x_lo == xmax else xmax - (xmax - x_lo) * scale
        return {"lower": lo, "upper": hi}

    def solve(lam, pole):
        """Find lam where LR(lam) = crit by Newton's method.

        `pole` is the singularity that lam must not cross (x_lo or x_hi).
        """
        for _ in range(50):
            # Compute LR and its derivative in a single pass
            s1 = s2 = 0.0
            for c, x in obs:
                t = 1.0 / (x - lam)
                s1 += c * t
                s2 += c * t * t
            nA = n / s1
            lr = 2.0 * sum(c * math.log((x - lam) / nA) for c, x in obs)
            dlr = 2.0 * (n * s2 - s1 * s1) / s1

            step = (lr - crit) / dlr
            lam -= step

            # Clamp: stay on the correct side of the pole
            if lam < pole:
                lam = min(lam, pole - 1e-12)
            else:
                lam = max(lam, pole + 1e-12)

            if abs(step) < 1e-12:
                break

        # Recover mu from the converged lambda
        return lam + n / s1

    lower = max(solve(x_lo - 1.0, x_lo), xmin)
    upper = min(solve(x_hi + 1.0, x_hi), xmax)

    return {"lower": lower, "upper": upper}
