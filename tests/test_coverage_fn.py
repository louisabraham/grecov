"""Tests for grecov_coverage."""

import pytest

from grecov.bfs import grecov_coverage
from grecov.solver import multinomial_ci


def _make_ci_fn(v, alpha, method="equal_tail"):
    """Return an interval function that computes a CI for each count vector."""

    def interval_fn(counts):
        result = multinomial_ci(list(counts), v, alpha=alpha, method=method)
        return result["lower"], result["upper"]

    return interval_fn


@pytest.mark.parametrize(
    "p,v,n",
    [
        ([0.5, 0.5], [0, 1], 10),
        ([0.3, 0.7], [0, 1], 16),
        ([0.25, 0.42, 0.33], [0, 1, 2], 12),
    ],
)
def test_coverage_at_least_nominal(p, v, n):
    """Coverage of a 95% CI should be >= 0.95 (exact CI is conservative)."""
    alpha = 0.05
    interval_fn = _make_ci_fn(v, alpha)
    result = grecov_coverage(p, v, n, interval_fn, eps=1e-4)

    assert result["coverage"] >= 1 - alpha - 1e-4, (
        f"coverage {result['coverage']:.6f} below nominal {1 - alpha}"
    )
    assert result["explored_mass"] > 0.999


def test_coverage_trivial_interval():
    """An interval covering the whole real line should have coverage ~1."""
    p = [0.3, 0.7]
    v = [0, 1]
    n = 10

    result = grecov_coverage(p, v, n, lambda _: (-1e10, 1e10), eps=1e-6)
    assert result["coverage"] == pytest.approx(result["explored_mass"], abs=1e-12)


def test_coverage_empty_interval():
    """An empty interval should have coverage 0."""
    p = [0.3, 0.7]
    v = [0, 1]
    n = 10

    result = grecov_coverage(p, v, n, lambda _: (1e10, 1e10), eps=1e-6)
    assert result["coverage"] == 0.0
