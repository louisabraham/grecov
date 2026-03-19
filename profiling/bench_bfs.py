"""Timing benchmarks for the C++ BFS hot path.

Uses p vectors extracted from a real solver run (multinomial_ci with
counts=[30,25,20,15,10], values=[0,1,2,3,4], alpha=0.05).

Benchmarks:
  --bfs       Single-call BFS microbenchmarks (tail + mass, varying n and p)
  --many      Many-call pattern simulating solver (buffer reuse, log table caching)
  --solver    Full solver end-to-end (equal_tail + mass methods)
  --all       All of the above (default)
"""

import argparse
import random
import timeit

from grecov._ext import grecov_tail, grecov_mass

# Shared parameters
v = [0, 1, 2, 3, 4]
eps = 5e-5  # eps_ratio=1e-3 * alpha=0.05
x_obs = [30, 25, 20, 15, 10]

# Probability vectors from solver extremes and common points
p_lower = [
    7.14093467e-01,
    1.36442907e-12,
    1.20061383e-12,
    9.95174051e-03,
    2.75954792e-01,
]
p_upper = [
    5.20119344e-01,
    1.05923135e-12,
    1.14109740e-02,
    1.34394004e-12,
    4.68469682e-01,
]
p_mle = [0.3, 0.25, 0.2, 0.15, 0.1]
p_uniform = [0.2, 0.2, 0.2, 0.2, 0.2]

cases = {
    "p_lower": p_lower,
    "p_upper": p_upper,
    "p_mle": p_mle,
    "p_uniform": p_uniform,
}


def _fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:>8.1f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:>8.1f} ms"
    else:
        return f"{seconds:>8.2f}  s"


def _auto_bench(fn, target_time=2.0):
    """Run fn enough times to get a stable measurement."""
    # Warm up
    fn()
    # Calibrate
    t = timeit.timeit(fn, number=1)
    n = max(1, int(target_time / t))
    total = timeit.timeit(fn, number=n)
    return total / n


def bench_tail_bfs():
    for n in [20, 50, 100]:
        s_obs = sum(ci * vi for ci, vi in zip(x_obs, v)) * n / 100
        print(f"\n=== grecov_tail (tail) | n={n}, k=5, eps={eps} ===")
        print(f"{'case':<12} {'time/call':>12} {'states':>10} {'M states/s':>12}")
        print("-" * 48)
        for name, p in cases.items():
            res = grecov_tail(p, v, s_obs, n, eps)
            t = _auto_bench(lambda p=p: grecov_tail(p, v, s_obs, n, eps))
            states = res["states_explored"]
            rate = states / t / 1e6 if t > 0 else 0
            print(f"{name:<12} {_fmt_time(t)} {states:>10} {rate:>11.2f}")


def bench_mass_bfs():
    for n_scale, x_obs_scaled in [(1, x_obs), (2, [c * 2 for c in x_obs])]:
        n = sum(x_obs_scaled)
        print(f"\n=== grecov_mass (mass) | n={n}, k=5, eps={eps} ===")
        print(f"{'case':<12} {'time/call':>12} {'states':>10} {'M states/s':>12}")
        print("-" * 48)
        for name, p in cases.items():
            res = grecov_mass(p, x_obs_scaled, eps, 1e-8)
            t = _auto_bench(lambda p=p, xo=x_obs_scaled: grecov_mass(p, xo, eps, 1e-8))
            states = res["states_explored"]
            rate = states / t / 1e6 if t > 0 and states > 0 else 0
            print(f"{name:<12} {_fmt_time(t)} {states:>10} {rate:>11.2f}")


def bench_many_calls():
    """Simulate solver pattern: many BFS calls with varying p, same n.

    Measures benefit of:
    - Cached log tables (log_fact, log_int rebuilt only when n changes)
    - Reusable hash set (cleared but not deallocated between calls)
    """
    print("\n=== Many-call pattern (simulating solver) ===")

    # Generate varied p vectors (like optimizer exploring the simplex)
    rng = random.Random(42)
    n_calls = 500
    p_vectors = []
    for _ in range(n_calls):
        raw = [rng.random() for _ in range(5)]
        s = sum(raw)
        p_vectors.append([x / s for x in raw])

    # Tail BFS many-call
    s_obs = sum(ci * vi for ci, vi in zip(x_obs, v))
    n = 100

    def run_tail_many():
        for p in p_vectors:
            grecov_tail(p, v, s_obs, n, eps)

    run_tail_many()  # warm up
    t_tail = timeit.timeit(run_tail_many, number=1)
    print(
        f"  tail BFS x{n_calls} (n={n}): {_fmt_time(t_tail)} total, {_fmt_time(t_tail / n_calls)}/call"
    )

    # Mass BFS many-call
    def run_mass_many():
        for p in p_vectors:
            grecov_mass(p, x_obs, eps, 1e-8)

    run_mass_many()
    t_mass = timeit.timeit(run_mass_many, number=1)
    print(
        f"  mass BFS x{n_calls} (n={n}): {_fmt_time(t_mass)} total, {_fmt_time(t_mass / n_calls)}/call"
    )


def bench_solver():
    from grecov.solver import multinomial_ci

    test_cases = [
        {
            "counts": [30, 25, 20, 15, 10],
            "values": [0, 1, 2, 3, 4],
            "label": "k=5 n=100",
        },
        {"counts": [15, 12, 10, 8, 5], "values": [0, 1, 2, 3, 4], "label": "k=5 n=50"},
        {"counts": [8, 6, 4, 2], "values": [0, 1, 2, 3], "label": "k=4 n=20"},
    ]

    for method in ["equal_tail", "greedy"]:
        print(f"\n=== Full solver (multinomial_ci, {method}) ===")
        print(f"{'case':<16} {'time':>12} {'bfs_calls':>10} {'bfs_states':>12}")
        print("-" * 52)
        for tc in test_cases:
            res = multinomial_ci(tc["counts"], tc["values"], alpha=0.05, method=method)
            t = _auto_bench(
                lambda tc=tc, m=method: multinomial_ci(
                    tc["counts"], tc["values"], alpha=0.05, method=m
                )
            )
            print(
                f"{tc['label']:<16} {_fmt_time(t)} {res['bfs_calls']:>10} {res['bfs_total_states']:>12}"
            )
            print(f"  CI: [{res['lower']:.6f}, {res['upper']:.6f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bfs", action="store_true", help="Run BFS microbenchmarks")
    parser.add_argument(
        "--many", action="store_true", help="Run many-call pattern benchmark"
    )
    parser.add_argument(
        "--solver", action="store_true", help="Run full solver benchmark"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.all or (not args.bfs and not args.many and not args.solver):
        args.bfs = args.many = args.solver = True

    if args.bfs:
        bench_tail_bfs()
        bench_mass_bfs()
    if args.many:
        bench_many_calls()
    if args.solver:
        bench_solver()
