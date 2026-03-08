"""
Implements the Neyman construction with the GreCov best-first enumeration algorithm.

Solves min/max v^T p  subject to  L(p) >= alpha/2  and  R(p) >= alpha/2,
where L(p) and R(p) are the two-sided tail probabilities computed by BFS.
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, TypeVar

import numpy as np

try:
    from cyipopt import minimize_ipopt
except ImportError:
    minimize_ipopt = None
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize

from grecov.bfs import grecov_bfs as _bfs_py
from grecov.bfs import grecov_mass_bfs as _cum_bfs_py

try:
    from grecov._ext import grecov_bfs as _bfs_ext  # type: ignore
    from grecov._ext import grecov_mass_bfs as _mass_bfs_ext  # type: ignore
except ImportError:
    logging.warning("Using Python implementation of BFS")
    _bfs_ext = None
    _mass_bfs_ext = None

logger = logging.getLogger("grecov")
logger.setLevel(logging.INFO)
logging.basicConfig()


def _softmax(theta_full):
    shift = theta_full - theta_full.max()
    e = np.exp(shift)
    return e / e.sum()


_T = TypeVar("_T")


def _cache_numpy(fn: Callable[[np.ndarray], _T]) -> Callable[[np.ndarray], _T]:
    """Cache decorator for functions taking a single numpy array argument."""
    _key: np.ndarray | None = None
    _result: _T | None = None

    @functools.wraps(fn)
    def wrapper(x: np.ndarray) -> _T:
        nonlocal _key, _result
        if _key is not None and np.array_equal(x, _key):
            return _result  # type: ignore[return-value]
        _result = fn(x)
        _key = np.array(x, dtype=np.float64)
        return _result

    return wrapper


# ── Parametrization ──────────────────────────────────────────────────────────

Param = Literal["direct", "reduced", "logit"]


class Parametrization(ABC):
    """Strategy for mapping optimizer variables to/from the simplex."""

    @abstractmethod
    def to_p(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def x0(self, p_init: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def bounds(self) -> list[tuple[float, float]]: ...

    @abstractmethod
    def tail_gradient(
        self, p: np.ndarray, n: int, prob: float, wsum: np.ndarray
    ) -> np.ndarray: ...

    @abstractmethod
    def tail_hessian(
        self,
        p: np.ndarray,
        n: int,
        prob: float,
        wsum: np.ndarray,
        wsum2: np.ndarray,
    ) -> np.ndarray: ...

    @abstractmethod
    def objective_jac(
        self, sign: float, v: np.ndarray, p: np.ndarray
    ) -> np.ndarray: ...

    @abstractmethod
    def objective_hess(
        self, sign: float, v: np.ndarray, p: np.ndarray, ndim: int
    ) -> np.ndarray: ...

    @abstractmethod
    def simplex_constraints(self, k: int, pmin: float) -> list[dict]: ...


class Direct(Parametrization):
    def __init__(self, k: int, pmin: float):
        self._k = k
        self._pmin = pmin

    def to_p(self, x):
        return x / x.sum()

    def x0(self, p_init):
        return p_init.copy()

    def bounds(self):
        return [(self._pmin, 1.0)] * self._k

    def tail_gradient(self, p, n, prob, wsum):
        return wsum / p

    def tail_hessian(self, p, n, prob, wsum, wsum2):
        return wsum2 / np.outer(p, p) - np.diag(wsum / (p * p))

    def objective_jac(self, sign, v, p):
        return sign * v

    def objective_hess(self, sign, v, p, ndim):
        return np.zeros((ndim, ndim))

    def simplex_constraints(self, k, pmin):
        return [
            {
                "type": "eq",
                "fun": lambda x: float(x.sum() - 1.0),
                "jac": lambda x: np.ones(k),
                "hess": lambda x, lam: np.zeros((k, k)),
            }
        ]


class Reduced(Parametrization):
    def __init__(self, k: int, pmin: float):
        self._k = k
        self._pmin = pmin

    def to_p(self, x):
        p_k = max(1.0 - x.sum(), 1e-300)
        return np.append(x, p_k)

    def x0(self, p_init):
        return p_init[:-1].copy()

    def bounds(self):
        r = self._k - 1
        return [(self._pmin, 1.0)] * r

    def tail_gradient(self, p, n, prob, wsum):
        return wsum[:-1] / p[:-1] - wsum[-1] / p[-1]

    def tail_hessian(self, p, n, prob, wsum, wsum2):
        H_full = wsum2 / np.outer(p, p) - np.diag(wsum / (p * p))
        K = self._k - 1
        return (
            H_full[:K, :K]
            - H_full[:K, K : K + 1]
            - H_full[K : K + 1, :K]
            + H_full[K, K]
        )

    def objective_jac(self, sign, v, p):
        return sign * (v[:-1] - v[-1])

    def objective_hess(self, sign, v, p, ndim):
        return np.zeros((ndim, ndim))

    def simplex_constraints(self, k, pmin):
        return [
            {
                "type": "ineq",
                "fun": lambda x: 1.0 - pmin - x.sum(),
                "jac": lambda x: -np.ones(k - 1),
                "hess": lambda x, lam: np.zeros((k - 1, k - 1)),
            }
        ]


class Logit(Parametrization):
    def __init__(self, k: int, theta_max: float):
        self._k = k
        self._theta_max = theta_max

    def to_p(self, x):
        theta_full = np.append(x, 0.0)
        return _softmax(theta_full)

    def x0(self, p_init):
        theta0 = np.log(p_init)
        return theta0[:-1] - theta0[-1]

    def bounds(self):
        r = self._k - 1
        return [(-self._theta_max, self._theta_max)] * r

    def tail_gradient(self, p, n, prob, wsum):
        return (wsum - n * prob * p)[:-1]

    def tail_hessian(self, p, n, prob, wsum, wsum2):
        r = self._k - 1
        H = wsum2[:r, :r].copy()
        H -= n * p[:r, None] * wsum[None, :r]
        H -= n * wsum[:r, None] * p[None, :r]
        H += n * (n + 1) * prob * np.outer(p[:r], p[:r])
        H -= n * prob * np.diag(p[:r])
        return H

    def objective_jac(self, sign, v, p):
        return (sign * p * (v - float(v @ p)))[:-1]

    def objective_hess(self, sign, v, p, ndim):
        k = self._k
        pr = p[: k - 1]
        d = pr * ((v - float(v @ p))[: k - 1])
        return sign * (np.diag(d) - np.outer(d, pr) - np.outer(pr, d))

    def simplex_constraints(self, k, pmin):
        return []  # softmax satisfies simplex automatically


def _make_param(name: Param, k: int, pmin: float, theta_max: float) -> Parametrization:
    if name == "direct":
        return Direct(k, pmin)
    elif name == "reduced":
        return Reduced(k, pmin)
    else:
        return Logit(k, theta_max)


# ── Public wrappers (for tests that import these) ────────────────────────────


def tail_gradient(
    p: np.ndarray, n: int, prob: float, wsum: np.ndarray, param: Param
) -> np.ndarray:
    """Gradient of a tail probability w.r.t. the chosen parametrization."""
    par = _make_param(param, len(p), 0.0, 0.0)
    return par.tail_gradient(p, n, prob, wsum)


def tail_hessian(
    p: np.ndarray,
    n: int,
    prob: float,
    wsum: np.ndarray,
    wsum2: np.ndarray,
    param: Param,
) -> np.ndarray:
    """Hessian of a tail probability w.r.t. the chosen parametrization."""
    par = _make_param(param, len(p), 0.0, 0.0)
    return par.tail_hessian(p, n, prob, wsum, wsum2)


# ── BFS result types ────────────────────────────────────────────────────────


class TailBFSResult(NamedTuple):
    prob_left: float
    prob_right: float
    wsum_left: np.ndarray
    wsum_right: np.ndarray
    wsum2_left: np.ndarray
    wsum2_right: np.ndarray


class EvalResult(NamedTuple):
    p: np.ndarray
    prob_left: float
    prob_right: float
    grad_left: np.ndarray
    grad_right: np.ndarray
    wsum_left: np.ndarray
    wsum_right: np.ndarray
    wsum2_left: np.ndarray
    wsum2_right: np.ndarray


# ── BFS stats accumulator ───────────────────────────────────────────────────


@dataclass
class BFSStats:
    calls: int = 0
    total_states: int = 0

    def record(self, states_explored: int) -> None:
        self.calls += 1
        self.total_states += states_explored


# ── Context dataclasses ──────────────────────────────────────────────────────


@dataclass
class TailContext:
    v: np.ndarray
    k: int
    n: int
    s_obs: float
    bound: float
    param: Parametrization
    optimizer: str
    opt_verbose: bool
    pmin: float
    eps: float
    x0: np.ndarray
    opt_bounds: list[tuple[float, float]]
    bfs_impl: Callable
    stats: BFSStats = field(default_factory=BFSStats)


@dataclass
class MassContext:
    v: np.ndarray
    k: int
    n: int
    bound: float
    param: Parametrization
    eps: float
    tie_margin: float
    x_obs_list: list[int]
    x0: np.ndarray
    opt_bounds: list[tuple[float, float]]
    bfs_impl: Callable
    stats: BFSStats = field(default_factory=BFSStats)


# ── BFS wrappers ─────────────────────────────────────────────────────────────


def _run_tail_bfs(p: np.ndarray, ctx: TailContext) -> TailBFSResult:
    bfs = ctx.bfs_impl(p.tolist(), ctx.v.tolist(), ctx.s_obs, ctx.n, ctx.eps)
    ctx.stats.record(bfs["states_explored"])
    k = ctx.k
    return TailBFSResult(
        prob_left=float(bfs["prob_left"]),
        prob_right=float(bfs["prob_right"]),
        wsum_left=np.asarray(bfs["wsum_left"], dtype=np.float64),
        wsum_right=np.asarray(bfs["wsum_right"], dtype=np.float64),
        wsum2_left=np.asarray(bfs["wsum2_left"], dtype=np.float64).reshape(k, k),
        wsum2_right=np.asarray(bfs["wsum2_right"], dtype=np.float64).reshape(k, k),
    )


def _run_mass_bfs(p: np.ndarray, ctx: MassContext) -> float:
    bfs = ctx.bfs_impl(p.tolist(), ctx.x_obs_list, ctx.eps, ctx.tie_margin)
    ctx.stats.record(bfs["states_explored"])
    return float(bfs["explored_mass"])


# ── Optimizer backends ────────────────────────────────────────────────────────


def _run_ipopt(
    objective, objective_jac, x0, constraints, opt_bounds, opt_verbose, obj_hess
):
    if minimize_ipopt is None:
        raise ImportError(
            "ipopt is not installed. Please install it with `pip install cyipopt`."
        )
    opts = {"disp": 5 if opt_verbose else 0, "maxiter": 200}
    result = minimize_ipopt(
        objective,
        x0,
        jac=objective_jac,
        hess=obj_hess,
        constraints=constraints,
        bounds=opt_bounds,
        options=opts,
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result


def _pack_trust_constr_constraints(constraints, ndim):
    """Convert dict-style constraints into trust-constr format.

    Inequality constraints that carry a ``"hess"`` key are merged into a
    single ``NonlinearConstraint``; all others are passed through with the
    ``"hess"`` key stripped.
    """
    ineq_hess = [c for c in constraints if c.get("type") == "ineq" and "hess" in c]
    other = [c for c in constraints if c not in ineq_hess]
    tc = []
    if ineq_hess:

        def _combined_fun(x, _cs=ineq_hess):
            return np.array([c["fun"](x) for c in _cs])

        def _combined_jac(x: np.ndarray, _cs: list[dict] = ineq_hess) -> np.ndarray:
            return np.vstack([c["jac"](x) for c in _cs])

        def _combined_hess(x, lam, _cs=ineq_hess):
            H = np.zeros((ndim, ndim))
            for i, c in enumerate(_cs):
                H += c["hess"](x, lam[i])
            return H

        tc.append(
            NonlinearConstraint(
                _combined_fun,
                0,
                np.inf,
                jac=_combined_jac,  # type: ignore[arg-type]
                hess=_combined_hess,
            )
        )
    for c in other:
        tc.append({ck: cv for ck, cv in c.items() if ck != "hess"})
    return tc


def _run_trust_constr(
    objective, objective_jac, x0, constraints, opt_bounds, opt_verbose, obj_hess
):
    tc_constraints = _pack_trust_constr_constraints(constraints, len(x0))
    result = minimize(
        objective,
        x0,
        method="trust-constr",
        jac=objective_jac,
        hess=obj_hess,
        constraints=tc_constraints,
        bounds=opt_bounds,
        options={"maxiter": 1000, "disp": opt_verbose},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result


def _run_slsqp(
    objective, objective_jac, x0, constraints, opt_bounds, opt_verbose, obj_hess
):
    """SLSQP backend with restart from best feasible point.

    SLSQP can overshoot into infeasible regions where constraint gradients
    vanish, causing it to get stuck.  To handle this, we track the best
    feasible point seen during optimization.  If SLSQP fails, we restart
    from that point (up to 3 attempts).  A single restart is usually
    sufficient since it places the initial point right at the constraint
    boundary.
    """
    slsqp_cons = [{k: v for k, v in c.items() if k != "hess"} for c in constraints]
    best_x = None
    best_fun = float("inf")

    def _tracking_objective(x):
        nonlocal best_x, best_fun
        f = objective(x)
        feasible = all(c["fun"](x) >= 0 for c in slsqp_cons if c["type"] == "ineq")
        if feasible and f < best_fun:
            best_x = x.copy()
            best_fun = f
        return f

    x = x0.copy()
    for _attempt in range(3):
        result = minimize(
            _tracking_objective,
            x,
            method="SLSQP",
            jac=objective_jac,
            constraints=slsqp_cons,
            bounds=opt_bounds,
            options={"maxiter": 1000, "ftol": 1e-10, "disp": opt_verbose},
        )
        if result.success:
            return result
        if best_x is not None:
            x = best_x
        else:
            break
    # Return best feasible point if we have one
    if best_x is not None:
        result.x = best_x
        result.fun = best_fun
        result.success = True
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result


_OPTIMIZER_BACKENDS = {
    "ipopt": _run_ipopt,
    "trust-constr": _run_trust_constr,
    "slsqp": _run_slsqp,
}


# ── Equal-tail endpoint solver ───────────────────────────────────────────────


def _build_tail_constraints(
    evaluate: Callable[[np.ndarray], EvalResult],
    param: Parametrization,
    ctx: TailContext,
) -> list[dict]:
    """Build optimizer constraints for the equal-tail problem."""
    n, k, bound = ctx.n, ctx.k, ctx.bound

    def _make_ineq(side: Literal["left", "right"]):
        prob_attr = f"prob_{side}"
        grad_attr = f"grad_{side}"
        wsum_attr = f"wsum_{side}"
        wsum2_attr = f"wsum2_{side}"

        def fun(x):
            return float(getattr(evaluate(x), prob_attr) - bound)

        def jac(x):
            return getattr(evaluate(x), grad_attr)

        def hess(x, lam):
            ev = evaluate(x)
            return lam * param.tail_hessian(
                ev.p,
                n,
                getattr(ev, prob_attr),
                getattr(ev, wsum_attr),
                getattr(ev, wsum2_attr),
            )

        return {"type": "ineq", "fun": fun, "jac": jac, "hess": hess}

    constraints = [_make_ineq("left"), _make_ineq("right")]
    constraints.extend(param.simplex_constraints(k, ctx.pmin))
    return constraints


def _solve_endpoint_tail(sign: float, ctx: TailContext) -> tuple[float, np.ndarray]:
    """Solve for one endpoint of the equal-tail confidence interval."""
    param, v, n = ctx.param, ctx.v, ctx.n
    ndim = len(ctx.x0)

    @_cache_numpy
    def evaluate(x: np.ndarray) -> EvalResult:
        p = param.to_p(x)
        bfs = _run_tail_bfs(p, ctx)
        return EvalResult(
            p=p,
            prob_left=bfs.prob_left,
            prob_right=bfs.prob_right,
            grad_left=param.tail_gradient(p, n, bfs.prob_left, bfs.wsum_left),
            grad_right=param.tail_gradient(p, n, bfs.prob_right, bfs.wsum_right),
            wsum_left=bfs.wsum_left,
            wsum_right=bfs.wsum_right,
            wsum2_left=bfs.wsum2_left,
            wsum2_right=bfs.wsum2_right,
        )

    def objective(x):
        return float(sign * v @ param.to_p(x))

    def objective_jac(x):
        return param.objective_jac(sign, v, evaluate(x).p)

    def obj_hess(x):
        return param.objective_hess(sign, v, evaluate(x).p, ndim)

    constraints = _build_tail_constraints(evaluate, param, ctx)

    backend = _OPTIMIZER_BACKENDS.get(ctx.optimizer)
    if backend is None:
        raise ValueError(f"Unknown optimizer: {ctx.optimizer!r}")
    result = backend(
        objective,
        objective_jac,
        ctx.x0.copy(),
        constraints,
        ctx.opt_bounds,
        ctx.opt_verbose,
        obj_hess,
    )

    # Feasibility check: verify both tail probabilities at the solution.
    # trust-constr can return success=True with violated constraints
    # (e.g. xtol termination with constr_violation > 0).
    p_sol = param.to_p(result.x)
    bfs_check = _run_tail_bfs(p_sol, ctx)
    tol = ctx.eps
    if bfs_check.prob_left < ctx.bound - tol or bfs_check.prob_right < ctx.bound - tol:
        raise RuntimeError(
            f"Optimizer returned infeasible point: "
            f"P_left={bfs_check.prob_left:.6g}, P_right={bfs_check.prob_right:.6g}, "
            f"bound={ctx.bound:.6g}"
        )

    return result.fun, p_sol


# ── Mass endpoint solver ─────────────────────────────────────────────────────


def _solve_endpoint_mass(sign: float, ctx: MassContext) -> tuple[float, np.ndarray]:
    param, v = ctx.param, ctx.v

    @_cache_numpy
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, float]:
        p = param.to_p(x)
        explored_mass = _run_mass_bfs(p, ctx)
        return p, explored_mass

    def objective(x):
        p = param.to_p(x)
        return float(sign * v @ p)

    def mass_fun(x):
        _, explored_mass = evaluate(x)
        return explored_mass

    constraint = NonlinearConstraint(mass_fun, 0.0, ctx.bound)
    result = differential_evolution(
        objective,
        ctx.opt_bounds,
        constraints=constraint,
        x0=ctx.x0.copy(),
        polish=False,
        popsize=10,
        tol=0.005,
        recombination=0.9,
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.fun, param.to_p(result.x)


# ── Public API ────────────────────────────────────────────────────────────────


def multinomial_ci(
    counts,
    values,
    alpha=0.05,
    *,
    method: Literal["equal_tail", "greedy"] = "equal_tail",
    eps_ratio: float = 1e-3,
    verbose=0,
    optimizer: str | None = None,
    param: str | None = None,
    pmin: float = 1e-8,
    theta_max: float = 10.0,
    tie_margin: float = 1e-8,
    use_python: bool = False,
):
    """Compute the confidence interval [lower, upper] for mu = v^T p.

    Parameters
    ----------
    counts : array-like of int
        Observed category counts.
    values : array-like of float
        Numerical value assigned to each category.
    alpha : float
        Significance level (default 0.05 for a 95% CI).
    method : str
        ``"equal_tail"`` (two-sided tail BFS) or ``"greedy"`` (likelihood-ordered BFS).
    eps_ratio : float
        BFS stopping tolerance as a fraction of alpha. Default: 1e-3.
    verbose : int
        Verbosity level: 0 = silent, 1 = BFS stats, 2 = optimizer output.
    optimizer : str or None
        Optimizer backend.  Default: ``"ipopt"`` or ``"trust-constr"``
        (depending on availability) for equal_tail, ``"de"`` for greedy.
    param : str or None
        Parametrization.  Default: depends on optimizer.
    pmin : float
        Minimum probability bound for direct/reduced parametrizations.
    theta_max : float
        Bound on theta parameters for softmax parametrization.
    tie_margin : float
        Log-probability margin for near-tie exclusion in BFS (greedy only).
    use_python : bool
        Force use of the pure-Python BFS implementation.

    Returns
    -------
    dict with keys: lower, upper, p_lower, p_upper, bfs_calls, bfs_total_states.
    """
    eps = eps_ratio * alpha

    counts = np.asarray(counts, dtype=int)
    v = np.asarray(values, dtype=float)
    k = len(v)
    n = int(counts.sum())

    assert k > 1, "k must be greater than 1"

    # -- Resolve defaults ------------------------------------------------------
    if method == "equal_tail":
        if optimizer is None:
            optimizer = "ipopt" if minimize_ipopt is not None else "slsqp"
        if param is None:
            param = {"ipopt": "direct", "trust-constr": "logit", "slsqp": "logit"}[
                optimizer
            ]
        bfs_impl = _bfs_py if use_python or _bfs_ext is None else _bfs_ext
    else:  # greedy
        if optimizer is None:
            optimizer = "de"
        if param is None:
            param = "reduced"
        bfs_impl = _cum_bfs_py if use_python or _mass_bfs_ext is None else _mass_bfs_ext

    par = _make_param(param, k, pmin, theta_max)  # type: ignore[arg-type]
    p_init = (counts + 0.5) / (n + 0.5 * k)
    p_init = p_init / p_init.sum()

    # -- Solve -----------------------------------------------------------------
    if method == "equal_tail":
        ctx = TailContext(
            v=v,
            k=k,
            n=n,
            s_obs=float(v @ counts),
            bound=alpha / 2.0,
            param=par,
            optimizer=optimizer,
            opt_verbose=(verbose >= 2),
            pmin=pmin,
            eps=eps,
            x0=par.x0(p_init),
            opt_bounds=par.bounds(),
            bfs_impl=bfs_impl,
        )
        lower, p_lower = _solve_endpoint_tail(+1.0, ctx)
        upper_neg, p_upper = _solve_endpoint_tail(-1.0, ctx)
        upper = -upper_neg
        stats = ctx.stats
    else:  # greedy
        ctx_mass = MassContext(
            v=v,
            k=k,
            n=n,
            bound=1.0 - alpha,
            param=par,
            eps=eps,
            tie_margin=tie_margin,
            x_obs_list=counts.tolist(),
            x0=par.x0(p_init),
            opt_bounds=par.bounds(),
            bfs_impl=bfs_impl,
        )
        lower, p_lower = _solve_endpoint_mass(+1.0, ctx_mass)
        upper_neg, p_upper = _solve_endpoint_mass(-1.0, ctx_mass)
        upper = -upper_neg
        stats = ctx_mass.stats

    if verbose >= 1:
        logger.info(
            f"upper={upper:.6f}  bfs_calls={stats.calls}  "
            f"states={stats.total_states}  "
            f"avg_states={stats.total_states / stats.calls if stats.calls > 0 else float('nan'):.2f}"
        )

    return {
        "lower": lower,
        "upper": upper,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "bfs_calls": stats.calls,
        "bfs_total_states": stats.total_states,
    }
