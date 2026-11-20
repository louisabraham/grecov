"""Calibrated Owen interval for the multinomial weighted mean.

Ordinary Owen inverts the profile LR with the universal Wilks cutoff
chi2_{1,1-alpha}.  It is very fast but undercovers badly.  Two separate
things go wrong, and both have to be fixed.

1. *Support.*  Owen's empirical likelihood maximises over the observed
   categories only, so its interval cannot leave the convex hull of the
   observed values.  When a category is unobserved the endpoint is
   pinned at the hull boundary and no cutoff, however large, can move
   it past a true mean that lies outside.  Measured at k=3, n=20: this
   alone caps the worst-case coverage at 0.4877 for every cutoff.
   The fix is to use the *full-simplex* profile LR (`_profile_lr` from
   `exact_profile_ci`), which lets zero-count categories take mass.
   That raises the uncalibrated floor to 0.8285.

2. *Cutoff.*  The remaining gap is closed by replacing the constant
   Wilks value with an observation-dependent cutoff

       I_c(x) = { z : T_z(x) <= c(x) },
       c(x)   = q * ( a + b / (1 + m(x)) ),   q = chi2_{1,1-alpha},

   with the normalised dispersion

       m(x) = n * sigma2_hat(x) / ((v_max - v_min)^2 / 4)   in [0, n].

   m = 0 for a degenerate sample and m ~ n for a spread one, so the
   cutoff is inflated exactly where the sample carries little
   information about the mean.

Online cost is O(k) for c(x) plus a scalar root find per endpoint.  The
price is offline: (a, b) must be calibrated per design.  DEFAULT_PARAMS
is calibrated for k=3, n=20, v=(0,1,2), alpha=0.05 and verified on a
20301-point lattice of the 2-simplex (boundary included); it is not a
general-k rule.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

from grecov.exact_profile_ci import _profile_lr
from grecov.owen_ci import profile_ci_crit

# (k, n) -> (a, b).  Grid-verified, not proved.
DEFAULT_PARAMS = {(3, 20): (1.18, 1.1)}

_EPS = 1e-9


def dispersion(counts, values):
    """m(x) = n * sigma2_hat / ((v_max - v_min)^2 / 4), in [0, n]."""
    c = np.asarray(counts, dtype=float)
    v = np.asarray(values, dtype=float)
    n = c.sum()
    if n <= 0:
        return 0.0
    ph = c / n
    mu = float(v @ ph)
    s2 = max(float((v**2) @ ph - mu**2), 0.0)
    span = float(v.max() - v.min())
    if span <= 0:
        return 0.0
    return n * s2 / (span**2 / 4.0)


def cutoff(counts, values, alpha, a, b):
    """Observation-dependent cutoff c(x)."""
    q = float(chi2.ppf(1.0 - alpha, df=1))
    return q * (a + b / (1.0 + dispersion(counts, values)))


def calibrated_owen_ci(counts, values, alpha=0.05, *, params=None):
    """Profile-LR interval with a calibrated, observation-dependent cutoff.

    Parameters
    ----------
    counts, values : array-like
    alpha : float
    params : (a, b) or None
        If None, looked up in DEFAULT_PARAMS by (k, n); raises for a
        design that has not been calibrated.

    Returns
    -------
    dict with keys: lower, upper.
    """
    c = np.asarray(counts, dtype=float)
    v = np.asarray(values, dtype=float)
    k = len(c)
    n = int(c.sum())
    if params is None:
        params = DEFAULT_PARAMS.get((k, n))
        if params is None:
            raise ValueError(
                f"no calibration for k={k}, n={n}; "
                "pass params=(a, b) or calibrate this design first"
            )
    a, b = params
    crit = cutoff(counts, values, alpha, a, b)

    # Fast path: with every category observed there is no zero-count
    # category to place mass on, so the full-simplex profile LR coincides
    # with Owen's, and Owen's analytic endpoint parameterisation applies
    # (two Newton solves instead of a bracketed root find).
    if np.all(c > 0):
        return profile_ci_crit(counts, values, crit)

    vmin, vmax = float(v.min()), float(v.max())
    muhat = float(v @ c / n)
    counts_l = list(np.asarray(counts, dtype=float))
    values_l = list(v)

    def f(z):
        return _profile_lr(counts_l, values_l, z) - crit

    lower = vmin if f(vmin + _EPS) <= 0 else brentq(f, vmin + _EPS, muhat, xtol=1e-7)
    upper = vmax if f(vmax - _EPS) <= 0 else brentq(f, muhat, vmax - _EPS, xtol=1e-7)
    return {"lower": float(lower), "upper": float(upper)}
