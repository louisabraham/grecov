"""Test that for k=2 the equal-tail CI matches Clopper-Pearson."""

import numpy as np
import pytest
from scipy.stats import beta

from grecov import confidence_interval


def clopper_pearson(x, n, alpha=0.05):
    """Clopper-Pearson exact binomial CI for proportion x/n."""
    if x == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, x, n - x + 1)
    if x == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, x + 1, n - x)
    return lower, upper


# (x, n) pairs to test
CASES = [
    (0, 10),
    (1, 10),
    (3, 10),
    (5, 10),
    (10, 10),
    (7, 20),
    (0, 5),
    (2, 5),
    (5, 5),
    (1, 30),
    (15, 30),
    (29, 30),
]

ALPHAS = [0.10, 0.05, 0.01, 0.001]

EPS_TOL = [
    (1e-3, 1e-3),
    (1e-4, 5e-3),
]


@pytest.mark.parametrize("x,n", CASES)
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("eps_ratio,tol", EPS_TOL)
def test_equal_tail_matches_clopper_pearson(x, n, alpha, eps_ratio, tol):
    cp_lower, cp_upper = clopper_pearson(x, n, alpha)

    counts = np.array([n - x, x])
    values = np.array([0.0, 1.0])

    result = confidence_interval(
        counts,
        values,
        alpha=alpha,
        method="equal_tail",
        eps_ratio=eps_ratio,
        use_python=True,
    )

    assert result["lower"] == pytest.approx(cp_lower, abs=tol), (
        f"lower: got {result['lower']}, expected {cp_lower}"
    )
    assert result["upper"] == pytest.approx(cp_upper, abs=tol), (
        f"upper: got {result['upper']}, expected {cp_upper}"
    )
