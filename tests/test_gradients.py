"""Test analytic gradients and hessians against scipy.differentiate."""

import numpy as np
import pytest

try:
    from scipy.differentiate import hessian, jacobian
except (ImportError, ModuleNotFoundError):
    pytest.skip("scipy.differentiate not available", allow_module_level=True)

from grecov.bfs import grecov_bfs as py_bfs
from grecov.solver import _softmax, tail_gradient, tail_hessian

try:
    from grecov._ext import grecov_bfs as cpp_bfs  # type: ignore
except ImportError:
    cpp_bfs = None

bfs_impl = cpp_bfs if cpp_bfs is not None else py_bfs


# ── Vectorized BFS wrappers for scipy.differentiate ─────────────────────────
# scipy.differentiate.jacobian/hessian expect f(x) where the first axis of x
# is the m-dimensional input vector, with extra batch dimensions appended.
# We use np.apply_along_axis to call the scalar BFS along axis 0.


def _make_prob_fn(v, s_obs, n, side, param):
    """Build a vectorized prob function for the given parametrization."""
    v_list = v.tolist()

    if param == "logit":

        def scalar_fn(th):
            th_full = np.append(th, 0.0)
            pp = _softmax(th_full)
            r = bfs_impl(pp.tolist(), v_list, float(s_obs), int(n), 1e-8)
            return float(r[f"prob_{side}"])
    else:  # reduced

        def scalar_fn(p_red):
            pp = np.append(p_red, max(1.0 - p_red.sum(), 1e-300))
            r = bfs_impl(pp.tolist(), v_list, float(s_obs), int(n), 1e-8)
            return float(r[f"prob_{side}"])

    def vectorized(x):
        return np.apply_along_axis(scalar_fn, axis=0, arr=x)

    return vectorized


# ── BFS result helper ────────────────────────────────────────────────────────


def _run_bfs(p, v, s_obs, n, eps=1e-8):
    r = bfs_impl(p.tolist(), v.tolist(), float(s_obs), int(n), eps)
    k = len(p)
    return {
        "prob_left": float(r["prob_left"]),
        "prob_right": float(r["prob_right"]),
        "wsum_left": np.asarray(r["wsum_left"], dtype=np.float64),
        "wsum_right": np.asarray(r["wsum_right"], dtype=np.float64),
        "wsum2_left": np.asarray(r["wsum2_left"], dtype=np.float64).reshape(k, k),
        "wsum2_right": np.asarray(r["wsum2_right"], dtype=np.float64).reshape(k, k),
    }


# ── Test cases ───────────────────────────────────────────────────────────────

CASES = [
    ([0.5, 0.5], [0.0, 1.0], [4, 12]),
    ([0.3, 0.7], [0.0, 1.0], [4, 12]),
    ([0.25, 0.42, 0.33], [0.0, 1.0, 2.0], [3, 5, 4]),
    ([0.14, 0.29, 0.36, 0.21], [0.0, 1.0, 2.0, 3.0], [2, 4, 5, 3]),
]


def _case_id(case):
    return f"k={len(case[0])}_n={sum(case[2])}"


def _get_x0(p, param):
    if param == "logit":
        theta_full = np.log(p)
        return (theta_full - theta_full[-1])[:-1]
    else:  # reduced
        return p[:-1].copy()


# ── Parametrized tests ───────────────────────────────────────────────────────


@pytest.mark.parametrize("p,v,counts", CASES, ids=[_case_id(c) for c in CASES])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("param", ["logit", "reduced"])
def test_gradient(p, v, counts, side, param):
    p = np.array(p)
    v = np.array(v)
    counts = np.array(counts)
    n = int(counts.sum())
    s_obs = float(v @ counts)
    x0 = _get_x0(p, param)

    bfs_result = _run_bfs(p, v, s_obs, n)
    grad_analytic = tail_gradient(
        p, n, bfs_result[f"prob_{side}"], bfs_result[f"wsum_{side}"], param
    )

    prob_fn = _make_prob_fn(v, s_obs, n, side, param)
    res = jacobian(prob_fn, x0, initial_step=1e-4, tolerances={"rtol": 1e-10})

    np.testing.assert_allclose(
        grad_analytic,
        res.df,
        atol=1e-8,
        rtol=1e-4,
        err_msg=f"Gradient mismatch ({side}, {param})",
    )


@pytest.mark.parametrize("p,v,counts", CASES, ids=[_case_id(c) for c in CASES])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("param", ["logit", "reduced"])
def test_hessian(p, v, counts, side, param):
    p = np.array(p)
    v = np.array(v)
    counts = np.array(counts)
    n = int(counts.sum())
    s_obs = float(v @ counts)
    x0 = _get_x0(p, param)

    bfs_result = _run_bfs(p, v, s_obs, n)
    hess_analytic = tail_hessian(
        p,
        n,
        bfs_result[f"prob_{side}"],
        bfs_result[f"wsum_{side}"],
        bfs_result[f"wsum2_{side}"],
        param,
    )

    prob_fn = _make_prob_fn(v, s_obs, n, side, param)
    res = hessian(prob_fn, x0, initial_step=0.5, step_factor=2.0)

    np.testing.assert_allclose(
        hess_analytic,
        res.ddf,
        atol=1e-6,
        rtol=1e-3,
        err_msg=f"Hessian mismatch ({side}, {param})",
    )
