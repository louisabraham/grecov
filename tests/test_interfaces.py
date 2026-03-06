"""Smoke tests that all public interfaces work without errors."""

import sys

import numpy as np
import pytest

from grecov.bfs import grecov_bfs, grecov_mass_bfs
from grecov.solver import multinomial_ci

_is_emscripten = sys.platform == "emscripten"


# ── BFS interfaces ──────────────────────────────────────────────────


def test_grecov_bfs_returns_expected_keys():
    result = grecov_bfs([0.3, 0.7], [0, 1], s_obs=8, n=10, eps=1e-4)
    expected_keys = {
        "prob_left",
        "prob_right",
        "wsum_left",
        "wsum_right",
        "wsum2_left",
        "wsum2_right",
        "explored_mass",
        "states_explored",
    }
    assert expected_keys == set(result.keys())


def test_grecov_mass_bfs_returns_expected_keys():
    result = grecov_mass_bfs([0.3, 0.7], [3, 7], eps=1e-3, tie_margin=1e-8)
    expected_keys = {"explored_mass", "states_explored"}
    assert expected_keys == set(result.keys())


def test_grecov_bfs_probs_in_range():
    result = grecov_bfs([0.25, 0.42, 0.33], [0, 1, 2], s_obs=5, n=6, eps=1e-4)
    assert 0.0 <= result["prob_left"] <= 1.0
    assert 0.0 <= result["prob_right"] <= 1.0
    assert 0.0 < result["explored_mass"] <= 1.0


def test_grecov_mass_bfs_mass_in_range():
    result = grecov_mass_bfs([0.25, 0.42, 0.33], [3, 5, 4], eps=1e-3)
    assert 0.0 <= result["explored_mass"] <= 1.0


# ── multinomial_ci interface ───────────────────────────────────


def test_ci_equal_tail_returns_expected_keys():
    result = multinomial_ci([3, 5, 4], [0, 1, 2], alpha=0.05, method="equal_tail")
    expected_keys = {
        "lower",
        "upper",
        "p_lower",
        "p_upper",
        "bfs_calls",
        "bfs_total_states",
    }
    assert expected_keys == set(result.keys())


def test_ci_greedy_returns_expected_keys():
    np.random.seed(42)
    result = multinomial_ci([3, 5, 4], [0, 1, 2], alpha=0.05, method="greedy")
    expected_keys = {
        "lower",
        "upper",
        "p_lower",
        "p_upper",
        "bfs_calls",
        "bfs_total_states",
    }
    assert expected_keys == set(result.keys())


def test_ci_equal_tail_lower_le_upper():
    result = multinomial_ci([3, 5, 4], [0, 1, 2], alpha=0.05, method="equal_tail")
    assert result["lower"] <= result["upper"]


def test_ci_greedy_lower_le_upper():
    np.random.seed(42)
    result = multinomial_ci([3, 5, 4], [0, 1, 2], alpha=0.05, method="greedy")
    assert result["lower"] <= result["upper"]


def test_ci_p_lower_is_valid_simplex():
    result = multinomial_ci([3, 5, 4], [0, 1, 2], alpha=0.05, method="equal_tail")
    p = result["p_lower"]
    assert len(p) == 3
    assert all(pi > 0 for pi in p)
    assert abs(sum(p) - 1.0) < 1e-4


def test_ci_equal_tail_use_python():
    result = multinomial_ci(
        [3, 5, 4], [0, 1, 2], alpha=0.05, method="equal_tail", use_python=True
    )
    assert result["lower"] <= result["upper"]


@pytest.mark.parametrize("param", ["direct", "reduced", "logit"])
def test_ci_equal_tail_parametrizations(param):
    optimizer = "trust-constr" if param == "logit" else "ipopt"
    try:
        result = multinomial_ci(
            [3, 5, 4],
            [0, 1, 2],
            alpha=0.05,
            method="equal_tail",
            param=param,
            optimizer=optimizer,
        )
        assert result["lower"] <= result["upper"]
    except ImportError:
        pytest.skip("ipopt not available")


def test_ci_k2():
    result = multinomial_ci([4, 12], [0, 1], alpha=0.05, method="equal_tail")
    assert result["lower"] <= result["upper"]
    assert 0.0 <= result["lower"]
    assert result["upper"] <= 1.0


# ── Error handling ─────────────────────────────────────────────────


def test_ci_k1_raises():
    with pytest.raises(AssertionError, match="k must be greater than 1"):
        multinomial_ci([10], [0], alpha=0.05, method="equal_tail")


@pytest.mark.skipif(_is_emscripten, reason="C++ exceptions crash Pyodide")
def test_ci_d9_equal_tail_raises():
    with pytest.raises(RuntimeError, match="dimension must be between 2 and 8"):
        multinomial_ci([1] * 9, list(range(9)), alpha=0.05, method="equal_tail")


@pytest.mark.skipif(_is_emscripten, reason="C++ exceptions crash Pyodide")
def test_ci_d9_greedy_raises():
    with pytest.raises(RuntimeError, match="dimension must be between 2 and 8"):
        multinomial_ci([1] * 9, list(range(9)), alpha=0.05, method="greedy")


@pytest.mark.skipif(_is_emscripten, reason="C++ exceptions crash Pyodide")
def test_ci_n_too_large_equal_tail_raises():
    with pytest.raises(RuntimeError, match="n must be <= 65535"):
        multinomial_ci([33000, 33000], [0, 1], alpha=0.05, method="equal_tail")


@pytest.mark.skipif(_is_emscripten, reason="C++ exceptions crash Pyodide")
def test_ci_n_too_large_greedy_raises():
    with pytest.raises(RuntimeError, match="n must be <= 65535"):
        multinomial_ci([33000, 33000], [0, 1], alpha=0.05, method="greedy")
