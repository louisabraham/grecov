"""Exact profile LR confidence interval via slice p-values.

p(z) = sup_{theta: v^T theta = z} P_theta(LR_z(X) <= LR_z(x_obs))
CI = {z : p(z) > alpha}

For fixed z the event A_z is fixed, so GreCov gradient/Hessian are valid.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from cyipopt import minimize_ipopt
except ImportError:  # optional dependency: pip install grecov[ipopt]
    minimize_ipopt = None

from grecov.bfs import grecov_iter

try:
    from grecov._ext import profile_bfs as _profile_bfs_ext
except ImportError:
    _profile_bfs_ext = None

_EPS = 1e-12


def _require_ipopt():
    if minimize_ipopt is None:
        raise ImportError(
            "ipopt is not installed. Please install it with `pip install cyipopt`."
        )


# ── Profile LR ──────────────────────────────────────────────────────────────


def _newton(f_df, x0, lo, hi, maxiter=60):
    """Find root of f on (lo, hi) via Newton with bisection safeguard."""
    x = x0
    for _ in range(maxiter):
        f, df = f_df(x)
        if abs(df) < _EPS:
            break
        prev = x
        x -= f / df
        if x <= lo:
            x = (prev + lo) / 2
        elif x >= hi:
            x = (prev + hi) / 2
        if abs(x - prev) < _EPS:
            break
    return x


def _profile_lr(counts, values, mu):
    """Profile LR statistic  -2 log(max_{q: v^Tq=mu} L(q) / max_q L(q)).

    Maximisation is over the full simplex: zero-count categories can
    receive mass to hit the target mean without affecting the likelihood.
    """
    vals = [float(v) for v in values]

    pos, zero_vs, n = [], [], 0.0
    for ci, vi in zip(counts, values):
        ci, vi = float(ci), float(vi)
        if ci > 0:
            pos.append((ci, vi - mu))
            n += ci
        else:
            zero_vs.append(vi)

    if n == 0:
        return math.inf
    # mu = sample mean → constrained MLE = unconstrained MLE → T = 0.
    # Check this before the support bounds so that mu at vmin/vmax is handled.
    mean_err = abs(sum(c * a for c, a in pos) / n)
    if mean_err < _EPS:
        return 0.0

    if not (min(vals) < mu < max(vals)):
        return math.inf

    lo, hi = -math.inf, math.inf
    for _, a in pos:
        if a > 0:
            lo = max(lo, -1.0 / a)
        elif a < 0:
            hi = min(hi, -1.0 / a)

    def lr(lam):
        return 2.0 * sum(c * math.log(1.0 + lam * a) for c, a in pos)

    obs_lo = min(a + mu for _, a in pos)
    obs_hi = max(a + mu for _, a in pos)

    # Interior solver: constrained MLE using only positive-count categories.
    best = math.inf
    if obs_lo < mu < obs_hi:

        def g(lam):
            s, ds = 0.0, 0.0
            for c, a in pos:
                d = 1.0 + lam * a
                s += (c / n) * a / d
                ds -= (c / n) * a * a / (d * d)
            return s, ds

        best = lr(_newton(g, 0.0, lo, hi))

    # Boundary solver: also try placing mass on zero-count categories.
    # This can give a LOWER T even when mu is interior to the observed support,
    # because the full-simplex constrained MLE may be better than the
    # positive-count-only one.
    if not zero_vs:
        return best
    targets = [v for v in zero_vs if v > obs_hi] + [v for v in zero_vs if v < obs_lo]
    if not targets and not math.isfinite(best):
        return math.inf

    for target in targets:

        def h(lam):
            S, M, dS, dM = 0.0, 0.0, 0.0, 0.0
            for c, a in pos:
                d = 1.0 + lam * a
                q, dq = c / (n * d), -c * a / (n * d * d)
                S += q
                M += q * (a + mu)
                dS += dq
                dM += dq * (a + mu)
            return (mu - M) - target * (1 - S), -dM + target * dS

        # Need lam that makes S < 1, freeing mass for zero-count categories.
        # target > mu: leak mass to high-value zeros (need to reduce positive mass)
        # target < mu: leak mass to low-value zeros
        # The sign of lam depends on the a_i values: to reduce S,
        # increase den = 1 + lam*a_i. For a_i < 0 (most common when
        # target is above observed support), lam < 0 increases den.
        sign = -1.0 if target > mu else 1.0
        lam0 = sign * 0.1
        for _ in range(100):
            if (
                all(1.0 + lam0 * a > _EPS for _, a in pos)
                and sum(c / (n * (1.0 + lam0 * a)) for c, a in pos) < 1 - _EPS
            ):
                break
            lam0 *= 0.5
        else:
            continue
        v = lr(_newton(h, lam0, lo, hi))
        if not math.isinf(v):
            best = min(best, v)
    return best


# ── Core: evaluate P_theta(A_z) with gradient and Hessian ────────────────────


def _f_grad_hess(theta, values, z, r_z, n, eps, counts_obs=None):
    """P_theta(A_z) and derivatives from a single BFS pass.

    A_z = {y : LR_z(y) <= r_z} is fixed for a given z.
    Uses C++ extension when available for ~10-50x speedup.
    """
    k = len(theta)
    t_cut = -2.0 * math.log(max(r_z, _EPS))

    if _profile_bfs_ext is not None and counts_obs is not None:
        # C++ computes t_cutoff internally from counts_obs,
        # ensuring the tie comparison uses the same floating-point path.
        res = _profile_bfs_ext(
            theta.tolist(), values.tolist(), z, [int(c) for c in counts_obs], n, eps
        )
        prob = res["prob"]
        wsum = np.asarray(res["wsum"], dtype=float)
        wsum2 = np.asarray(res["wsum2"], dtype=float).reshape(k, k)
    else:
        # Compute t_obs directly from counts_obs (not via r_z round-trip)
        # to ensure the tie comparison uses the same floating-point path.
        t_obs_direct = (
            _profile_lr(counts_obs, values, z) if counts_obs is not None else t_cut
        )
        prob, wsum, wsum2 = 0.0, np.zeros(k), np.zeros((k, k))
        for X, _lp, p in grecov_iter(theta.tolist(), n, eps=eps):
            if _profile_lr(X, values, z) >= t_obs_direct:
                prob += p
                x = np.array(X, dtype=float)
                wsum += p * x
                wsum2 += p * np.outer(x, x)

    th = np.maximum(theta, _EPS)
    grad = wsum / th
    hess = wsum2 / np.outer(th, th)
    np.fill_diagonal(hess, (np.diag(wsum2) - wsum) / th**2)
    return prob, grad, hess


# ── Helpers ──────────────────────────────────────────────────────────────────


def _project_to_slice(theta, values, z):
    """Shift theta so that v^T theta = z, staying near the simplex."""
    a = values - z
    err = float(theta @ a)
    ac = a - a.mean()
    d = float(ac @ ac)
    if d > _EPS:
        theta = theta - (err / d) * ac
    theta = np.clip(theta, _EPS, None)
    return theta / theta.sum()


def _constrained_mle(phat, values, z):
    """Constrained MLE with v^T theta = z."""
    a = values - z
    if abs(float(phat @ a)) < 1e-10:
        return phat.copy()
    lo, hi = -math.inf, math.inf
    for ai in a:
        if ai > 0:
            lo = max(lo, -1.0 / ai)
        elif ai < 0:
            hi = min(hi, -1.0 / ai)

    def g(lam):
        s = sum(p * ai / (1 + lam * ai) for p, ai in zip(phat, a))
        ds = -sum(p * ai**2 / (1 + lam * ai) ** 2 for p, ai in zip(phat, a))
        return s, ds

    lam = _newton(g, 0.0, lo, hi)
    theta = phat / (1.0 + lam * a)
    return np.clip(theta, _EPS, None) / np.clip(theta, _EPS, None).sum()


# ── p(z) = sup_{v^T theta = z} P_theta(A_z) ─────────────────────────────────


def p_value(z, counts, values, eps=1e-6, warm_start=None):
    """Compute p(z) and the optimal theta (for warm-starting the next call)."""
    _require_ipopt()
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    k = len(values)
    n = int(counts.sum())
    phat = counts / n

    t_obs = _profile_lr(counts, values, z)
    r_z = math.exp(-t_obs / 2) if not math.isinf(t_obs) else 0.0

    # Single-entry cache: IPOPT calls f, grad, hess at the same point.
    _last_key = None
    _last_val = None

    counts_obs_list = counts.tolist()

    def _eval(th):
        nonlocal _last_key, _last_val
        key = th.tobytes()
        if key != _last_key:
            _last_key = key
            _last_val = _f_grad_hess(
                np.asarray(th), values, z, r_z, n, eps, counts_obs=counts_obs_list
            )
        return _last_val

    # Starting points: constrained MLE + support-2 vertices + warm start.
    # Support-2 vertices are the extreme points of the slice polytope:
    # for each pair (i,j), theta_i v_i + theta_j v_j = z with theta_i + theta_j = 1.
    # The global optimum is often at or near a vertex.
    starts = []
    try:
        starts.append(_constrained_mle(phat, values, z))
    except Exception:
        starts.append(phat)
    for i in range(k):
        for j in range(i + 1, k):
            if abs(values[i] - values[j]) < _EPS:
                continue
            t = (z - values[j]) / (values[i] - values[j])
            if -0.01 < t < 1.01:  # include near-boundary vertices
                t = max(_EPS, min(1 - _EPS, t))
                th = np.full(k, _EPS)
                th[i] = t
                th[j] = 1 - t
                starts.append(th / th.sum())
    if warm_start is not None:
        starts.append(_project_to_slice(np.asarray(warm_start, dtype=float), values, z))

    ones_k = np.ones(k)
    zero_h = np.zeros((k, k))
    ipopt_args = dict(
        bounds=[(_EPS, 1.0)] * k,
        constraints=[
            {
                "type": "eq",
                "fun": lambda th: float(th.sum() - 1.0),
                "jac": lambda th: ones_k,
                "hess": lambda th, v: zero_h,
            },
            {
                "type": "eq",
                "fun": lambda th: float(values @ th - z),
                "jac": lambda th: values,
                "hess": lambda th, v: zero_h,
            },
        ],
        options={
            "print_level": 0,
            "sb": "yes",
            "max_iter": 50,
            "tol": 1e-8,
            "acceptable_tol": 1e-6,
            "constr_viol_tol": 1e-10,
        },
    )

    best, best_theta = 0.0, starts[0]
    for x0 in starts:
        try:
            res = minimize_ipopt(
                lambda th: -_eval(th)[0],
                x0,
                jac=lambda th: -_eval(th)[1],
                hess=lambda th: -_eval(th)[2],
                **ipopt_args,
            )
            if -res.fun > best:
                best, best_theta = -res.fun, res.x
        except Exception:
            pass
    return best, best_theta


# ── CI = {z : p(z) > alpha} via bisection with warm-starting ────────────────


def profile_ci(counts, values, alpha=0.05, eps=1e-6, eta_z=1e-3):
    """Exact profile likelihood confidence interval.

    Parameters
    ----------
    counts : array-like of int
        Observed category counts.
    values : array-like of float
        Numerical value assigned to each category.
    alpha : float
        Significance level (default 0.05 for a 95% CI).
    eps : float
        BFS truncation threshold.
    eta_z : float
        Bisection tolerance on z.

    Returns
    -------
    dict with keys: lower, upper.
    """
    from grecov.owen_ci import profile_ci as owen_ci

    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    muhat = float(values @ counts / counts.sum())
    vmin, vmax = float(values.min()), float(values.max())

    # Use Owen's asymptotic CI to narrow the bisection bracket.
    # The exact CI is typically within 50% of the gap from asymptotic to boundary.
    owen = owen_ci(counts.tolist(), values.tolist(), alpha)
    gap_lo = owen["lower"] - vmin
    gap_hi = vmax - owen["upper"]
    bracket_lo = max(vmin + 1e-6, owen["lower"] - 0.5 * gap_lo)
    bracket_hi = min(vmax - 1e-6, owen["upper"] + 0.5 * gap_hi)
    # muhat can sit on the support edge (a degenerate sample), where the
    # 1e-6 clamps would put the bracket on the far side of it and return
    # an interval that excludes the sample mean.
    bracket_lo = min(bracket_lo, muhat)
    bracket_hi = max(bracket_hi, muhat)

    def bisect(lo, hi, left):
        prev = None
        while hi - lo > eta_z:
            mid = (lo + hi) / 2
            pv, prev = p_value(mid, counts, values, eps, prev)
            if pv > alpha:
                if left:
                    hi = mid
                else:
                    lo = mid
            else:
                if left:
                    lo = mid
                else:
                    hi = mid
        # Return the rejected side: the endpoint must lie outside the
        # accepted set, otherwise the interval is up to eta_z too narrow
        # and coverage drops below 1 - alpha.
        return lo if left else hi

    lower = bisect(bracket_lo, muhat, left=True)
    upper = bisect(muhat, bracket_hi, left=False)
    return {"lower": lower, "upper": upper}
