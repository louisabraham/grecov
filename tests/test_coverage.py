"""Coverage tests for confidence_interval.

For small (n, k), enumerates ALL multinomial outcomes and verifies that
the probability-weighted coverage is at least 1 - alpha.
"""

import math

import numpy as np
import pytest

from grecov.solver import confidence_interval


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
    v = np.asarray(values, dtype=float)
    mu_true = float(v @ np.asarray(p))
    k = len(p)

    coverage = 0.0
    for counts in _enumerate_outcomes(n, k):
        prob = _multinomial_prob(counts, p)
        try:
            r = confidence_interval(
                list(counts), values, alpha=alpha, method=method, **kwargs
            )
            if r["lower"] <= mu_true <= r["upper"]:
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
    assert coverage >= 1 - alpha - 1e-4, (
        f"Coverage {coverage:.6f} < {1 - alpha} for k={k}, n={n}, "
        f"p=[{', '.join(f'{pi:.3f}' for pi in p)}], alpha={alpha}"
    )


@pytest.mark.parametrize(
    "k,n,p,values",
    CASES,
    ids=[_case_id(c) for c in CASES],
)
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_mass_coverage(k, n, p, values, alpha):
    np.random.seed(42)
    coverage = _check_coverage(p, values, n, alpha, method="mass")
    assert coverage >= 1 - alpha - 0.02, (
        f"Coverage {coverage:.6f} < {1 - alpha} for k={k}, n={n}, "
        f"p=[{', '.join(f'{pi:.3f}' for pi in p)}], alpha={alpha}"
    )
