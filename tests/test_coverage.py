"""Coverage tests for multinomial_ci and grecov_coverage.

For small (n, k), enumerates ALL multinomial outcomes and verifies that
the probability-weighted coverage is at least 1 - alpha.  Also checks
that grecov_coverage (BFS-based approximation) matches the exact
enumeration to within the BFS stopping tolerance.
"""

import math

import numpy as np
import pytest

from grecov.bfs import grecov_coverage
from grecov.solver import multinomial_ci

# ── Helpers ──────────────────────────────────────────────────────────────────


def _enumerate_outcomes(n, k):
    """Yield all k-tuples of non-negative integers summing to n."""
    if k == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in _enumerate_outcomes(n - first, k - 1):
            yield (first, *rest)


def _multinomial_prob(x, p):
    """Multinomial probability P(x | p)."""
    n = sum(x)
    log_prob = math.lgamma(n + 1)
    for xi, pi in zip(x, p):
        log_prob -= math.lgamma(xi + 1)
        if xi > 0:
            log_prob += xi * math.log(max(pi, 1e-300))
    return math.exp(log_prob)


def _check_coverage(p, values, n, alpha, method, **kwargs):
    """Compute exact coverage by enumerating all multinomial outcomes.

    Returns the total probability mass of outcomes whose CI covers the
    true mean mu = v^T p.
    """

    def interval_fn(counts):
        return multinomial_ci(
            list(counts), values, alpha=alpha, method=method, **kwargs
        )

    return _exact_coverage(p, values, n, interval_fn)


def _exact_coverage(p, values, n, interval_fn):
    """Compute exact coverage by enumerating all multinomial outcomes.

    interval_fn maps a count tuple to a dict with "lower"/"upper" keys
    or a (lower, upper) tuple.
    """
    v = np.asarray(values, dtype=float)
    mu_true = float(v @ np.asarray(p))
    k = len(p)

    coverage = 0.0
    for counts in _enumerate_outcomes(n, k):
        prob = _multinomial_prob(counts, p)
        try:
            interval = interval_fn(counts)
            if isinstance(interval, dict):
                lo, hi = interval["lower"], interval["upper"]
            else:
                lo, hi = interval
            if lo <= mu_true <= hi:
                coverage += prob
        except Exception:
            # Optimization failure — conservatively count as covered
            coverage += prob

    return coverage


def _random_p(k, rng):
    """Sample a random probability vector from Dirichlet(1,...,1)."""
    return rng.dirichlet(np.ones(k)).tolist()


# ── Generate test cases with fixed seed ──────────────────────────────────────

_RNG = np.random.RandomState(12345)

CASES = []
for k in [2, 3, 4]:
    for n in [4, 6]:
        CASES.append((k, n, _random_p(k, _RNG), list(range(k))))


def _case_id(case):
    return f"k={case[0]}_n={case[1]}"


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "k,n,p,values",
    CASES,
    ids=[_case_id(c) for c in CASES],
)
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_equal_tail_coverage(k, n, p, values, alpha):
    coverage = _check_coverage(p, values, n, alpha, method="equal_tail")
    assert coverage >= 1 - alpha, (
        f"Coverage {coverage:.6f} < {1 - alpha} for k={k}, n={n}, "
        f"p=[{', '.join(f'{pi:.3f}' for pi in p)}], alpha={alpha}"
    )


@pytest.mark.parametrize(
    "k,n,p,values",
    CASES,
    ids=[_case_id(c) for c in CASES],
)
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_equal_tail_coverage_trust_constr(k, n, p, values, alpha):
    coverage = _check_coverage(
        p, values, n, alpha, method="equal_tail", optimizer="trust-constr"
    )
    assert coverage >= 1 - alpha, (
        f"Coverage {coverage:.6f} < {1 - alpha} for k={k}, n={n}, "
        f"p=[{', '.join(f'{pi:.3f}' for pi in p)}], alpha={alpha}"
    )


@pytest.mark.parametrize(
    "k,n,p,values",
    CASES,
    ids=[_case_id(c) for c in CASES],
)
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_greedy_coverage(k, n, p, values, alpha):
    np.random.seed(42)
    coverage = _check_coverage(p, values, n, alpha, method="greedy")
    assert coverage >= 1 - alpha, (
        f"Coverage {coverage:.6f} < {1 - alpha} for k={k}, n={n}, "
        f"p=[{', '.join(f'{pi:.3f}' for pi in p)}], alpha={alpha}"
    )


# ── grecov_coverage vs exact enumeration ─────────────────────────────────────


COVERAGE_CASES = [
    (k, n, _random_p(k, np.random.RandomState(99)), list(range(k)))
    for k in [2, 3]
    for n in [4, 6]
]


@pytest.mark.parametrize(
    "k,n,p,values",
    COVERAGE_CASES,
    ids=[f"k={c[0]}_n={c[1]}" for c in COVERAGE_CASES],
)
def test_grecov_coverage_matches_exact(k, n, p, values):
    """grecov_coverage should match full enumeration within BFS tolerance."""
    alpha = 0.05
    eps = 1e-6

    def interval_fn(counts):
        return multinomial_ci(list(counts), values, alpha=alpha)

    exact = _exact_coverage(p, values, n, interval_fn)
    approx = grecov_coverage(p, values, n, interval_fn, eps=eps)

    assert approx["coverage"] == pytest.approx(exact, abs=eps), (
        f"grecov_coverage={approx['coverage']:.8f} vs exact={exact:.8f}"
    )
    assert approx["explored_mass"] > 1.0 - eps
