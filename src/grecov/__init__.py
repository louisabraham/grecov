"""grecov — Exact confidence intervals for multinomial distributions."""

__version__ = "0.2.0"

from grecov.bfs import grecov_iter, grecov_mass, grecov_tail
from grecov.jeffreys_ci import jeffreys_ci
from grecov.owen_ci import profile_ci
from grecov.gaffke_ci import gaffke_ci
from grecov.owen_buehler_ci import owen_buehler_ci
from grecov.calibrated_owen_ci import calibrated_owen_ci
from grecov.exact_profile_ci import profile_ci as exact_profile_ci
from grecov.solver import multinomial_ci

__all__ = [
    "multinomial_ci",
    "jeffreys_ci",
    "profile_ci",
    "gaffke_ci",
    "owen_buehler_ci",
    "calibrated_owen_ci",
    "exact_profile_ci",
    "grecov_tail",
    "grecov_mass",
    "grecov_iter",
    "__version__",
]
