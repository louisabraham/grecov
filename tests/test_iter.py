"""Tests for grecov_iter."""

import math

import pytest

from grecov.bfs import grecov_iter, grecov_tail

CASES = [
    ([0.5, 0.5], 16),
    ([0.3, 0.7], 16),
    ([0.25, 0.42, 0.33], 12),
    ([0.30, 0.40, 0.30], 30),
    ([0.14, 0.29, 0.36, 0.21], 14),
    ([0.15, 0.30, 0.35, 0.20], 40),
    ([0.10, 0.25, 0.30, 0.20, 0.15], 20),
    ([0.4, 0.6], 500),
    ([0.3, 0.7], 1000),
    ([0.30, 0.40, 0.30], 500),
]


@pytest.mark.parametrize("p,n", CASES)
def test_yields_decreasing_probability(p, n):
    prev = float("inf")
    for _counts, _log_p, prob in grecov_iter(p, n, eps=1e-4):
        assert prob <= prev + 1e-15
        prev = prob


@pytest.mark.parametrize("p,n", CASES)
def test_log_prob_consistent_with_prob(p, n):
    for _counts, log_p, prob in grecov_iter(p, n, eps=1e-4):
        assert math.exp(log_p) == pytest.approx(prob, rel=1e-12)


@pytest.mark.parametrize("p,n", CASES)
def test_counts_sum_to_n(p, n):
    for counts, _log_p, _prob in grecov_iter(p, n, eps=1e-4):
        assert sum(counts) == n


@pytest.mark.parametrize("p,n", CASES)
def test_mass_matches_grecov_tail(p, n):
    v = list(range(len(p)))
    s_obs = sum(c * vi for c, vi in zip([0] * len(p), v))
    eps = 1e-6

    ref = grecov_tail(p, v, s_obs, n, eps)

    enum_mass = 0.0
    num_states = 0
    for _counts, _log_p, prob in grecov_iter(p, n, eps):
        enum_mass += prob
        num_states += 1

    assert num_states == ref["states_explored"]
    assert enum_mass == pytest.approx(ref["explored_mass"], abs=1e-12)


@pytest.mark.parametrize("p,n", CASES)
def test_no_duplicate_states(p, n):
    seen = set()
    for counts, _log_p, _prob in grecov_iter(p, n, eps=1e-4):
        assert counts not in seen, f"duplicate: {counts}"
        seen.add(counts)


def test_first_yield_is_mode(p=[0.3, 0.7], n=10):
    """The first yielded state should have the highest probability."""
    items = list(grecov_iter(p, n, eps=1e-6))
    first_prob = items[0][2]
    assert all(prob <= first_prob + 1e-15 for _, _, prob in items)
