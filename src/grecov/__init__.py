"""grecov — Exact confidence intervals for multinomial distributions."""

__version__ = "0.2.0"

from grecov.bfs import grecov_iter, grecov_mass, grecov_tail
from grecov.jeffreys_ci import jeffreys_ci
from grecov.profile_ci import profile_ci
from grecov.solver import multinomial_ci

__all__ = [
    "multinomial_ci",
    "jeffreys_ci",
    "profile_ci",
    "grecov_tail",
    "grecov_mass",
    "grecov_iter",
    "__version__",
]
