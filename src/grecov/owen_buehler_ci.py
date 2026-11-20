"""Owen-ordered Buehler confidence interval for the multinomial mean.

The Buehler construction takes an ordering statistic r(y) and returns

    L(x) = inf{ z : p_x(z) > alpha/2 },
    p_x(z) = sup_{p : v^T p <= z} P_p(E_x),   E_x = {y : r(y) >= r(x)}.

The classical (equal-tail / sum-ordered) choice is r(y) = v^T y.  Here
r(y) is instead the *Owen profile-LR lower endpoint* of y, which
separates outcomes with the same empirical mean but different
dispersion.  The chi2_1 cutoff is used only to define the ordering, so
the coverage of the result does not depend on the Wilks approximation.

Two properties make this well behaved:

* E_x is fixed once x is observed, so P_p(E_x) is a smooth function of
  p and the enumeration engine supplies its gradient and Hessian.
* the feasible sets {p : v^T p <= z} are nested in z, so p_x is
  non-decreasing and the inversion is monotone.  There is no
  disconnected-root problem, unlike a moving LR ordering.

The upper limit is the lower limit of the reflected problem (values
negated), which keeps the two endpoints exactly symmetric in code.
"""

from __future__ import annotations


import numpy as np

try:
    from cyipopt import minimize_ipopt
except ImportError:  # optional dependency: pip install grecov[ipopt]
    minimize_ipopt = None

from grecov.bfs import grecov_iter
from grecov.owen_ci import profile_ci as _owen

_EPS = 1e-12


def _require_ipopt():
    if minimize_ipopt is None:
        raise ImportError(
            "ipopt is not installed. Please install it with `pip install cyipopt`."
        )


def _order_stat(counts, values, alpha, cache):
    """r(y) = Owen lower endpoint of y, memoised on the count vector."""
    hit = cache.get(counts)
    if hit is None:
        hit = _owen(list(counts), values, alpha)["lower"]
        cache[counts] = hit
    return hit


def _f_grad_hess(theta, values, r_obs, n, eps, alpha, cache):
    """P_theta(E_x) and its first two derivatives from one enumeration pass."""
    k = len(theta)
    prob = 0.0
    wsum = np.zeros(k)
    wsum2 = np.zeros((k, k))
    for X, _lp, p in grecov_iter(theta.tolist(), n, eps=eps):
        if _order_stat(X, values, alpha, cache) >= r_obs:
            prob += p
            x = np.asarray(X, dtype=float)
            wsum += p * x
            wsum2 += p * np.outer(x, x)
    th = np.maximum(theta, _EPS)
    grad = wsum / th
    hess = wsum2 / np.outer(th, th)
    np.fill_diagonal(hess, (np.diag(wsum2) - wsum) / th**2)
    return prob, grad, hess


def _starts(values, z, k, phat):
    """Vertices of {p : v^T p <= z} intersect simplex, plus a feasible interior point."""
    out = []
    # simplex vertices that satisfy the half-space
    for i in range(k):
        if values[i] <= z + 1e-12:
            th = np.full(k, _EPS)
            th[i] = 1.0
            out.append(th / th.sum())
    # support-two points on the boundary v^T p = z
    for i in range(k):
        for j in range(i + 1, k):
            if abs(values[i] - values[j]) < _EPS:
                continue
            t = (z - values[j]) / (values[i] - values[j])
            if -0.01 < t < 1.01:
                t = min(max(t, _EPS), 1 - _EPS)
                th = np.full(k, _EPS)
                th[i] = t
                th[j] = 1 - t
                out.append(th / th.sum())
    # observed frequencies, pulled into the half-space if needed
    q = np.clip(phat, _EPS, None)
    q = q / q.sum()
    if float(values @ q) > z:
        lo = np.zeros(k)
        lo[int(np.argmin(values))] = 1.0
        a, b = 0.0, 1.0
        for _ in range(60):
            m = (a + b) / 2
            if float(values @ ((1 - m) * q + m * lo)) > z:
                a = m
            else:
                b = m
        q = np.clip((1 - b) * q + b * lo, _EPS, None)
        q = q / q.sum()
    out.append(q)
    return out


def _p_value(z, counts, values, alpha, n, eps, cache, warm=None):
    """sup over the half-space {v^T p <= z} of P_p(E_x)."""
    _require_ipopt()
    k = len(values)
    r_obs = _order_stat(tuple(int(c) for c in counts), values, alpha, cache)
    phat = np.asarray(counts, float) / n

    last = {}

    def ev(th):
        key = th.tobytes()
        if key not in last:
            last.clear()
            last[key] = _f_grad_hess(
                np.asarray(th), values, r_obs, n, eps, alpha, cache
            )
        return last[key]

    ones = np.ones(k)
    zero_h = np.zeros((k, k))
    cons = [
        {
            "type": "eq",
            "fun": lambda t: float(t.sum() - 1.0),
            "jac": lambda t: ones,
            "hess": lambda t, v_: zero_h,
        },
        {
            "type": "ineq",
            "fun": lambda t: float(z - values @ t),
            "jac": lambda t: -values,
            "hess": lambda t, v_: zero_h,
        },
    ]
    opts = {
        "print_level": 0,
        "sb": "yes",
        "max_iter": 50,
        "tol": 1e-8,
        "acceptable_tol": 1e-6,
        "constr_viol_tol": 1e-10,
    }

    starts = _starts(values, z, k, phat)
    if warm is not None:
        w = np.clip(np.asarray(warm, float), _EPS, None)
        w = w / w.sum()
        if float(values @ w) <= z + 1e-12:
            starts.append(w)

    best, best_th = 0.0, starts[0]
    for x0 in starts:
        # a start already gives a valid lower bound on the sup
        v0 = ev(np.asarray(x0, float))[0]
        if v0 > best:
            best, best_th = v0, np.asarray(x0, float)
        try:
            res = minimize_ipopt(
                lambda t: -ev(t)[0],
                x0,
                jac=lambda t: -ev(t)[1],
                hess=lambda t: -ev(t)[2],
                bounds=[(_EPS, 1.0)] * k,
                constraints=cons,
                options=opts,
            )
            if -res.fun > best and float(values @ res.x) <= z + 1e-8:
                best, best_th = -res.fun, res.x
        except Exception:
            pass
    return best, best_th


def _lower(counts, values, alpha, eps, eta_z):
    """inf{ z : p_x(z) > alpha/2 } by bisection (p_x is monotone in z)."""
    counts = np.asarray(counts, float)
    values = np.asarray(values, float)
    n = int(counts.sum())
    cache = {}
    lo, hi = float(values.min()), float(values @ counts / n)
    target = alpha / 2.0

    if _p_value(lo, counts, values, alpha, n, eps, cache)[0] > target:
        return lo
    warm = None
    while hi - lo > eta_z:
        mid = (lo + hi) / 2
        pv, warm = _p_value(mid, counts, values, alpha, n, eps, cache, warm)
        if pv > target:
            hi = mid
        else:
            lo = mid
    return lo  # rejected side: keeps the interval on the conservative side


def owen_buehler_ci(counts, values, alpha=0.05, *, eps=1e-6, eta_z=1e-3):
    """Owen-ordered Buehler confidence interval for mu = v^T p.

    Returns
    -------
    dict with keys: lower, upper.
    """
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    lower = _lower(counts, values, alpha, eps, eta_z)
    upper = -_lower(counts, -values, alpha, eps, eta_z)
    return {"lower": float(lower), "upper": float(upper)}
