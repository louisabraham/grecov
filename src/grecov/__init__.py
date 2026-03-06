"""grecov — Exact confidence intervals for multinomial distributions."""

__version__ = "0.2.0"

from grecov.bfs import grecov_bfs, grecov_mass_bfs
from grecov.solver import multinomial_ci

__all__ = ["multinomial_ci", "grecov_bfs", "grecov_mass_bfs", "__version__"]
