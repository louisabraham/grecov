"""grecov — Exact confidence intervals for multinomial distributions."""

__version__ = "0.2.0"

from grecov.bfs import grecov_iter, grecov_mass, grecov_tail
from grecov.solver import multinomial_ci

__all__ = ["multinomial_ci", "grecov_tail", "grecov_mass", "grecov_iter", "__version__"]
