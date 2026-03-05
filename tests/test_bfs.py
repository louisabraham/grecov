"""Tests that Python and C++ BFS implementations produce identical results."""

import pytest

from grecov._ext import grecov_bfs as cpp_bfs  # type: ignore
from grecov._ext import grecov_mass_bfs as cpp_mass_bfs  # type: ignore
from grecov.bfs import grecov_bfs as py_bfs
from grecov.bfs import grecov_mass_bfs as py_mass_bfs

# ── Test cases ───────────────────────────────────────────────────────

BFS_CASES = [
    # (p, v, counts)
    ([0.5, 0.5], [0, 1], [4, 12]),
    ([0.3, 0.7], [0, 1], [4, 12]),
    ([0.25, 0.42, 0.33], [0, 1, 2], [3, 5, 4]),
    ([0.30, 0.40, 0.30], [0, 1, 2], [8, 12, 10]),
    ([0.14, 0.29, 0.36, 0.21], [0, 1, 2, 3], [2, 4, 5, 3]),
    ([0.15, 0.30, 0.35, 0.20], [0, 1, 2, 3], [6, 12, 14, 8]),
    ([0.10, 0.25, 0.30, 0.20, 0.15], [0, 1, 2, 3, 4], [2, 5, 6, 4, 3]),
    # Large n (k=2, n=500)
    ([0.4, 0.6], [0, 1], [200, 300]),
    # Large n (k=2, n=1000)
    ([0.3, 0.7], [0, 1], [300, 700]),
    # Large n (k=3, n=500)
    ([0.30, 0.40, 0.30], [0, 1, 2], [150, 200, 150]),
    # Large n (k=3, n=1000)
    ([0.25, 0.42, 0.33], [0, 1, 2], [250, 420, 330]),
]

MASS_CASES = [
    # (p, x_obs)
    ([0.5, 0.5], [4, 12]),
    ([0.3, 0.7], [4, 12]),
    ([0.25, 0.42, 0.33], [3, 5, 4]),
    ([0.30, 0.40, 0.30], [8, 12, 10]),
    ([0.20, 0.20, 0.20, 0.20, 0.20], [2, 5, 6, 4, 3]),
    ([0.14, 0.29, 0.36, 0.21], [2, 4, 5, 3]),
    ([0.05, 0.30, 0.35, 0.15, 0.15], [2, 5, 6, 4, 3]),
    # Near-uniform (many near-ties)
    ([0.20, 0.20, 0.20, 0.20, 0.20], [4, 4, 4, 4, 4]),
    # Skewed
    ([0.02, 0.40, 0.40, 0.10, 0.08], [2, 5, 6, 4, 3]),
    # Large n (k=2, n=500)
    ([0.4, 0.6], [200, 300]),
    # Large n (k=2, n=1000)
    ([0.3, 0.7], [300, 700]),
    # Large n (k=3, n=500)
    ([0.30, 0.40, 0.30], [150, 200, 150]),
    # Large n (k=3, n=1000)
    ([0.25, 0.42, 0.33], [250, 420, 330]),
]


# ── grecov_bfs tests ────────────────────────────────────────────────


@pytest.mark.parametrize("p,v,counts", BFS_CASES)
def test_grecov_bfs_match(p, v, counts):
    n = sum(counts)
    s_obs = sum(c * vi for c, vi in zip(counts, v))
    eps = 1e-6

    r_py = py_bfs(p, v, s_obs, n, eps)
    r_cpp = cpp_bfs(p, v, s_obs, n, eps)

    assert r_py["states_explored"] == r_cpp["states_explored"], (
        f"states_explored mismatch: py={r_py['states_explored']} cpp={r_cpp['states_explored']}"
    )
    # Larger n accumulates more floating-point error; scale tolerance accordingly
    tol = max(1e-12, n * 1e-13)
    tol2 = max(1e-10, n * n * 1e-13)
    assert r_py["prob_left"] == pytest.approx(r_cpp["prob_left"], abs=tol)
    assert r_py["prob_right"] == pytest.approx(r_cpp["prob_right"], abs=tol)
    assert r_py["explored_mass"] == pytest.approx(r_cpp["explored_mass"], abs=tol)
    k = len(p)
    for i in range(k):
        assert r_py["wsum_left"][i] == pytest.approx(r_cpp["wsum_left"][i], abs=tol)
        assert r_py["wsum_right"][i] == pytest.approx(r_cpp["wsum_right"][i], abs=tol)
    for idx in range(k * k):
        assert r_py["wsum2_left"][idx] == pytest.approx(
            r_cpp["wsum2_left"][idx], abs=tol2
        ), (
            f"wsum2_left[{idx}] mismatch: py={r_py['wsum2_left'][idx]} cpp={r_cpp['wsum2_left'][idx]}"
        )
        assert r_py["wsum2_right"][idx] == pytest.approx(
            r_cpp["wsum2_right"][idx], abs=tol2
        ), (
            f"wsum2_right[{idx}] mismatch: py={r_py['wsum2_right'][idx]} cpp={r_cpp['wsum2_right'][idx]}"
        )


# ── grecov_mass_bfs tests ───────────────────────────────────────────


@pytest.mark.parametrize("p,x_obs", MASS_CASES)
def test_grecov_mass_bfs_match(p, x_obs):
    eps = 1e-3
    tie_margin = 1e-8

    r_py = py_mass_bfs(p, x_obs, eps, tie_margin)
    r_cpp = cpp_mass_bfs(p, x_obs, eps, tie_margin)

    assert r_py["states_explored"] == r_cpp["states_explored"], (
        f"states_explored mismatch: py={r_py['states_explored']} cpp={r_cpp['states_explored']}"
    )
    assert r_py["explored_mass"] == pytest.approx(r_cpp["explored_mass"], abs=1e-12)


@pytest.mark.parametrize("tie_margin", [1e-2, 1e-4, 1e-8, 1e-12])
def test_mass_bfs_tie_margins(tie_margin):
    """Different tie margins should produce identical results between Python and C++."""
    p = [0.20, 0.20, 0.20, 0.20, 0.20]
    x_obs = [2, 5, 6, 4, 3]

    r_py = py_mass_bfs(p, x_obs, eps=1e-3, tie_margin=tie_margin)
    r_cpp = cpp_mass_bfs(p, x_obs, eps=1e-3, tie_margin=tie_margin)

    assert r_py["states_explored"] == r_cpp["states_explored"]
    assert r_py["explored_mass"] == pytest.approx(r_cpp["explored_mass"], abs=1e-12)


def test_mass_bfs_at_mode():
    """When x_obs is the mode, mass should be 0 (no states strictly above)."""
    p = [0.25, 0.42, 0.33]
    x_obs = [3, 5, 4]

    r_py = py_mass_bfs(p, x_obs)
    r_cpp = cpp_mass_bfs(p, x_obs)

    assert r_py["explored_mass"] == 0.0
    assert r_cpp["explored_mass"] == 0.0
    assert r_py["states_explored"] == 1
    assert r_cpp["states_explored"] == 1
