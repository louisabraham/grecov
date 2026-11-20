"""Tests for jeffreys_ci and profile_ci."""

import numpy as np
import pytest

from grecov.jeffreys_ci import jeffreys_ci
from grecov.owen_ci import profile_ci

# Cases including many zeros, small n, degenerate, and normal
CASES = [
    ([3, 5, 4], [0, 1, 2]),
    ([4, 12], [0, 1]),
    ([25, 30, 35, 10], [0, 1, 2, 3]),
    ([10, 20, 30, 40], [0, 1, 2, 3]),
    ([1, 1, 1, 1], [0, 1, 2, 3]),
    # Zero-count categories
    ([50, 50, 0, 0], [0, 1, 2, 3]),
    ([0, 0, 50, 50], [0, 1, 2, 3]),
    ([0, 30, 0, 70], [0, 1, 2, 3]),
    ([5, 5, 0, 0, 0, 0, 0, 90], list(range(8))),
    ([1, 0, 0, 0, 0, 0, 0, 9], list(range(8))),
    ([20, 20, 20, 20, 20, 0, 0, 0], list(range(8))),
    ([1, 1, 1, 1, 1, 1, 1, 1], list(range(8))),
    # Degenerate: single category
    ([100, 0, 0, 0], [0, 1, 2, 3]),
    ([0, 0, 0, 100], [0, 1, 2, 3]),
    # Near-degenerate
    ([1, 0, 0, 99], [0, 1, 2, 3]),
    ([1, 1, 1, 97], [0, 1, 2, 3]),
    # Small n
    ([1, 0, 0, 0, 0, 0, 0, 9], list(range(8))),
    ([0, 0, 0, 0, 0, 0, 0, 10], list(range(8))),
]


@pytest.mark.parametrize("counts,values", CASES)
def test_jeffreys_ci_valid(counts, values):
    r = jeffreys_ci(counts, values)
    assert r["lower"] <= r["upper"]
    assert r["lower"] >= min(values)
    assert r["upper"] <= max(values)


@pytest.mark.parametrize("counts,values", CASES)
def test_profile_ci_valid(counts, values):
    r = profile_ci(counts, values)
    assert r["lower"] <= r["upper"]
    assert r["lower"] >= min(values)
    assert r["upper"] <= max(values)


@pytest.mark.parametrize("counts,values", CASES)
def test_jeffreys_ci_contains_mle(counts, values):
    c = np.array(counts, dtype=float)
    v = np.array(values, dtype=float)
    n = c.sum()
    if n == 0:
        return
    # Jeffreys prior pulls the posterior away from the boundary,
    # so the MLE may not be contained when all mass is on one category.
    n_positive = sum(1 for ci in counts if ci > 0)
    if n_positive < 2:
        return
    mle = float(v @ c / n)
    r = jeffreys_ci(counts, values, alpha=0.01)
    assert r["lower"] <= mle <= r["upper"]


@pytest.mark.parametrize("counts,values", CASES)
def test_profile_ci_contains_mle(counts, values):
    c = np.array(counts, dtype=float)
    v = np.array(values, dtype=float)
    n = c.sum()
    if n == 0:
        return
    mle = float(v @ c / n)
    r = profile_ci(counts, values, alpha=0.01)
    assert r["lower"] <= mle <= r["upper"]


def test_jeffreys_narrower_at_larger_n():
    v = [0, 1, 2]
    r_small = jeffreys_ci([3, 5, 4], v)
    r_large = jeffreys_ci([30, 50, 40], v)
    assert (r_large["upper"] - r_large["lower"]) < (r_small["upper"] - r_small["lower"])


def test_profile_narrower_at_larger_n():
    v = [0, 1, 2]
    r_small = profile_ci([3, 5, 4], v)
    r_large = profile_ci([30, 50, 40], v)
    assert (r_large["upper"] - r_large["lower"]) < (r_small["upper"] - r_small["lower"])


@pytest.mark.parametrize("method", [jeffreys_ci, profile_ci])
def test_alpha_monotone(method):
    counts = [10, 20, 30, 40]
    values = [0, 1, 2, 3]
    r_wide = method(counts, values, alpha=0.01)
    r_narrow = method(counts, values, alpha=0.10)
    assert r_wide["lower"] <= r_narrow["lower"]
    assert r_wide["upper"] >= r_narrow["upper"]
