"""
Implements the Neyman construction with the GreCov best-first enumeration algorithm.

Solves min/max v^T p  subject to  L(p) >= alpha/2  and  R(p) >= alpha/2,
where L(p) and R(p) are the two-sided tail probabilities computed by BFS.
"""

import logging
import time

import numpy as np
from scipy.optimize import Bounds, minimize

try:
    from grecov._ext import grecov_bfs as _bfs_impl
except ImportError:
    logging.warning("Using Python implementation of BFS")
    from grecov.bfs import grecov_bfs as _bfs_impl


def _softmax(theta_full):
    shift = theta_full - theta_full.max()
    e = np.exp(shift)
    return e / e.sum()


def confidence_interval(counts, values, alpha=0.05, eps=1e-6, verbose=False):
    """Compute the confidence interval [lower, upper] for mu = v^T p.

    Parameters
    ----------
    counts : array-like of int
        Observed category counts.
    values : array-like of float
        Numerical value assigned to each category.
    alpha : float
        Significance level (default 0.05 for a 95% CI).
    eps : float
        BFS stopping tolerance.
    verbose : bool
        Print optimizer progress.

    Returns
    -------
    dict with keys: lower, upper, time_seconds.
    """
    t0 = time.perf_counter()

    counts = np.asarray(counts, dtype=int)
    v = np.asarray(values, dtype=float)
    k = len(v)
    n = int(counts.sum())
    s_obs = float(v @ counts)

    assert k > 1, "k must be greater than 1"

    # Initialise from observed frequencies with pseudo-count
    p_init = (counts + 0.5) / (n + 0.5 * k)
    p_init /= p_init.sum()
    theta0 = np.log(p_init)
    theta0 = (theta0 - theta0[-1])[:-1]

    bound = alpha / 2.0
    log_bound = float(np.log(bound))
    # [-10, 10] in logit space covers p_i in ~[0.00002, 0.99995]
    theta_bounds = Bounds(np.full(k - 1, -10.0), np.full(k - 1, 10.0))

    def solve_endpoint(sign):
        # SLSQP queries the same theta multiple times per iteration
        # (objective, jacobian, constraints), so we cache the last result.
        cached_key = None
        cached_result = None

        def evaluate(theta_red):
            nonlocal cached_key, cached_result

            key = theta_red.astype(np.float64).tobytes()
            if key == cached_key:
                return cached_result

            theta_full = np.append(theta_red, 0.0)
            p = _softmax(theta_full)

            bfs = _bfs_impl(p.tolist(), v.tolist(), s_obs, n, eps)

            prob_left = float(bfs["prob_left"])
            prob_right = float(bfs["prob_right"])
            wsum_left = np.asarray(bfs["wsum_left"], dtype=np.float64)
            wsum_right = np.asarray(bfs["wsum_right"], dtype=np.float64)

            # Gradient in reduced theta-space (chain rule through softmax)
            grad_left = (wsum_left - n * prob_left * p)[:-1]
            grad_right = (wsum_right - n * prob_right * p)[:-1]

            cached_key = key
            cached_result = {
                "p": p,
                "prob_left": prob_left,
                "prob_right": prob_right,
                "grad_left": grad_left,
                "grad_right": grad_right,
            }
            return cached_result

        def objective(theta):
            return float(sign * v @ evaluate(theta)["p"])

        def objective_jac(theta):
            res = evaluate(theta)
            p = res["p"]
            mu = float(v @ p)
            return (sign * p * (v - mu))[:-1]

        def scale_constraint(prob, grad):
            safe = max(prob, 1e-300)
            return float(np.log(safe) - log_bound), grad / safe

        def make_constraint(side):
            def fun(theta):
                res = evaluate(theta)
                return scale_constraint(res[f"prob_{side}"], res[f"grad_{side}"])[0]

            def jac(theta):
                res = evaluate(theta)
                return scale_constraint(res[f"prob_{side}"], res[f"grad_{side}"])[1]

            return {"type": "ineq", "fun": fun, "jac": jac}

        constraints = [make_constraint("left"), make_constraint("right")]

        result = minimize(
            objective,
            theta0,
            method="SLSQP",
            jac=objective_jac,
            constraints=constraints,
            bounds=theta_bounds,
            options={"disp": verbose, "ftol": 1e-6, "maxiter": 200},
        )

        final = evaluate(result.x)
        constraints_ok = (
            final["prob_left"] >= bound - 1e-4 and final["prob_right"] >= bound - 1e-4
        )

        if not (result.success and constraints_ok):
            side = "lower" if sign > 0 else "upper"
            raise RuntimeError(f"Optimization failed for {side} bound")

        return float(v @ final["p"])

    lower = solve_endpoint(+1.0)
    upper = solve_endpoint(-1.0)

    return {
        "lower": lower,
        "upper": upper,
        "time_seconds": time.perf_counter() - t0,
    }
