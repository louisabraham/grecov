"""Coverage comparison of Jeffreys and Profile CI methods.

For random probability vectors p, computes exact coverage
P_p({x : mu ∈ CI(x)}) using grecov_iter, for k=3,4 and n=100,200,500.

Uses multiprocessing to parallelize across p vectors.
"""

import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from grecov.bfs import grecov_iter
from grecov.jeffreys_ci import jeffreys_ci
from grecov.owen_ci import profile_ci

try:
    get_ipython()  # type: ignore[name-defined]
except NameError:
    plt.switch_backend("Agg")

# ── Parameters ────────────────────────────────────────────────────────────────

ALPHA = 0.05
EPS = 1e-4
SEED = 42

CONFIGS = [
    (3, [10, 20], 20),
    (4, [10, 20], 20),
]

# ── Worker function ───────────────────────────────────────────────────────────


def _compute_coverage(args):
    """Compute coverage of both methods for a single (p, n) pair."""
    p_list, v, n, alpha, eps = args
    target = sum(pi * vi for pi, vi in zip(p_list, v))

    cov_j = 0.0
    cov_p = 0.0
    mass = 0.0
    states = 0

    for counts, _log_p, prob in grecov_iter(p_list, n, eps):
        states += 1
        mass += prob

        r_j = jeffreys_ci(list(counts), v, alpha=alpha)
        if r_j["lower"] <= target <= r_j["upper"]:
            cov_j += prob

        r_p = profile_ci(list(counts), v, alpha=alpha)
        if r_p["lower"] <= target <= r_p["upper"]:
            cov_p += prob

    return {
        "cov_jeffreys": cov_j,
        "cov_profile": cov_p,
        "explored_mass": mass,
        "states": states,
    }


# ── Generate random p vectors ─────────────────────────────────────────────────

rng = np.random.default_rng(SEED)


def random_simplex(k, n_samples):
    raw = rng.exponential(1.0, size=(n_samples, k))
    return raw / raw.sum(axis=1, keepdims=True)


# ── Run coverage experiments ──────────────────────────────────────────────────

if __name__ == "__main__":
    results = []
    n_workers = min(mp.cpu_count(), 8)

    for k, n_values, n_random in CONFIGS:
        v = list(range(k))
        p_vectors = random_simplex(k, n_random)

        for n in n_values:
            tasks = [(p.tolist(), v, n, ALPHA, EPS) for p in p_vectors]

            with mp.Pool(n_workers) as pool:
                raw = list(
                    tqdm(
                        pool.imap(_compute_coverage, tasks),
                        total=len(tasks),
                        desc=f"k={k} n={n}",
                    )
                )

            for p_idx, (p, r) in enumerate(zip(p_vectors, raw)):
                results.append(
                    {
                        "k": k,
                        "n": n,
                        "p_idx": p_idx,
                        "target": float(p @ np.arange(k)),
                        **r,
                    }
                )

    # ── Plots ─────────────────────────────────────────────────────────────────

    all_k = sorted(set(c[0] for c in CONFIGS))
    all_n = sorted(set(n for _, ns, _ in CONFIGS for n in ns))

    fig, axes = plt.subplots(
        len(all_k),
        len(all_n),
        figsize=(5 * len(all_n), 4.5 * len(all_k)),
        squeeze=False,
    )

    nominal = 1 - ALPHA

    for row, k in enumerate(all_k):
        for col, n in enumerate(all_n):
            ax = axes[row, col]

            subset = [r for r in results if r["k"] == k and r["n"] == n]
            if not subset:
                ax.set_visible(False)
                continue

            targets = [r["target"] for r in subset]
            cov_j = [r["cov_jeffreys"] for r in subset]
            cov_p = [r["cov_profile"] for r in subset]

            order = np.argsort(targets)
            targets_sorted = [targets[i] for i in order]
            cov_j_sorted = [cov_j[i] for i in order]
            cov_p_sorted = [cov_p[i] for i in order]

            ax.plot(
                targets_sorted, cov_j_sorted, "o-", ms=4, label="Jeffreys", alpha=0.8
            )
            ax.plot(
                targets_sorted,
                cov_p_sorted,
                "s-",
                ms=4,
                label="Profile LR",
                alpha=0.8,
            )
            ax.axhline(nominal, color="red", ls="--", lw=1, label=f"nominal {nominal}")
            ax.set_ylim(nominal - 0.06, 1.005)
            ax.set_xlabel(r"$\mu = v^\top p$")
            ax.set_ylabel("Coverage")
            ax.set_title(f"k={k}, n={n}")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Exact coverage of Jeffreys vs Profile LR ({int(100 * (1 - ALPHA))}% CI)\n"
        rf"random p vectors, $\alpha={ALPHA}$",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig("exps/coverage_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to exps/coverage_comparison.png")

    # ── Summary table ─────────────────────────────────────────────────────────

    print(
        f"\n{'k':>3} {'n':>5} {'#p':>4} "
        f"{'Jeff min':>10} {'Jeff mean':>10} "
        f"{'Prof min':>10} {'Prof mean':>10} "
        f"{'states':>8}"
    )
    print("-" * 72)
    for k, n_values, _ in CONFIGS:
        for n in n_values:
            subset = [r for r in results if r["k"] == k and r["n"] == n]
            if not subset:
                continue
            cj = [r["cov_jeffreys"] for r in subset]
            cp = [r["cov_profile"] for r in subset]
            avg_states = int(np.mean([r["states"] for r in subset]))
            print(
                f"{k:>3} {n:>5} {len(subset):>4} "
                f"{min(cj):>10.6f} {np.mean(cj):>10.6f} "
                f"{min(cp):>10.6f} {np.mean(cp):>10.6f} "
                f"{avg_states:>8}"
            )
