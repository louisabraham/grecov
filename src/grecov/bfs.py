"""
Python implementation of the GreCov BFS algorithm.

Best-first enumeration of multinomial count vectors in decreasing
probability order. Returns two-sided tail probabilities and the
conditional expectations needed for analytic gradients.
"""

import heapq
import math


def _start_counts(p, n):
    """Balanced rounding of n*p to integers."""
    k = len(p)
    floors = [int(math.floor(pi * n)) for pi in p]
    fracs = [pi * n - f for pi, f in zip(p, floors)]
    remainder = n - sum(floors)
    indices = sorted(range(k), key=lambda i: fracs[i], reverse=True)
    for r in range(remainder):
        floors[indices[r]] += 1
    return floors


def grecov_bfs(p, v, s_obs, n, eps=1e-4):
    """
    Parameters
    ----------
    p : list of float
        Probability vector.
    v : list of float
        Value vector for each category.
    s_obs : float
        Observed test statistic v^T x.
    n : int
        Total count (sum of observations).
    eps : float
        Stopping tolerance on unexplored mass.

    Returns
    -------
    dict with keys:
        prob_left, prob_right    — tail probabilities P(S <= s_obs), P(S >= s_obs)
        wsum_left, wsum_right — sum_x P(x)*x for each tail
        explored_mass            — total probability mass visited
        states_explored          — number of states popped from the heap
    """
    assert n > 0, "n must be positive"

    k = len(p)

    MIN_P = 1e-300
    p_stable = [max(pi, MIN_P) for pi in p]
    p_sum = sum(p_stable)
    p_stable = [pi / p_sum for pi in p_stable]

    log_p = [math.log(pi) for pi in p_stable]

    # Precompute log-factorials
    log_fact = [0.0] * (n + 1)
    for i in range(1, n + 1):
        log_fact[i] = log_fact[i - 1] + math.log(i)

    def log_prob(counts):
        val = log_fact[n]
        for ci, lp in zip(counts, log_p):
            val -= log_fact[ci]
            val += ci * lp
        return val

    # Start from the approximate mode
    start = tuple(_start_counts(p_stable, n))
    heap = [(-log_prob(start), start)]  # min-heap with negated keys = max-heap
    visited = {start}

    mass = 0.0
    # Per-tail accumulators: 0=left, 1=right, 2=equal
    probs = [0.0, 0.0, 0.0]
    wsums = [[0.0] * k for _ in range(3)]
    wsum2s = [[0.0] * (k * k) for _ in range(3)]
    states_explored = 0

    def accumulate(side, p_state, counts):
        probs[side] += p_state
        ws = wsums[side]
        ws2 = wsum2s[side]
        for i in range(k):
            pci = p_state * counts[i]
            ws[i] += pci
            for j in range(i, k):
                val = pci * counts[j]
                ws2[i * k + j] += val
                if i != j:
                    ws2[j * k + i] += val

    L, R, E = 0, 1, 2

    while heap:
        neg_lp, counts = heapq.heappop(heap)
        log_p_state = -neg_lp
        states_explored += 1

        p_state = math.exp(log_p_state)
        mass += p_state

        # Classify: s < s_obs (left), s > s_obs (right), s == s_obs (both)
        s_val = sum(ci * vi for ci, vi in zip(counts, v))

        if s_val < s_obs:
            accumulate(L, p_state, counts)
        elif s_val > s_obs:
            accumulate(R, p_state, counts)
        else:
            accumulate(E, p_state, counts)

        if 1.0 - eps <= mass:
            break

        # Generate neighbours: transfer one unit from j to i
        for j in range(k):
            if counts[j] == 0:
                continue
            for i in range(k):
                if i == j:
                    continue
                neighbor = list(counts)
                neighbor[i] += 1
                neighbor[j] -= 1
                neighbor = tuple(neighbor)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                log_p_neighbor = (
                    log_p_state
                    + math.log(counts[j])
                    - math.log(counts[i] + 1)
                    + log_p[i]
                    - log_p[j]
                )
                heapq.heappush(heap, (-log_p_neighbor, neighbor))

    # States with s == s_obs contribute to both tails
    return {
        "prob_left": probs[L] + probs[E],
        "prob_right": probs[R] + probs[E],
        "wsum_left": [wsums[L][i] + wsums[E][i] for i in range(k)],
        "wsum_right": [wsums[R][i] + wsums[E][i] for i in range(k)],
        "wsum2_left": [wsum2s[L][i] + wsum2s[E][i] for i in range(k * k)],
        "wsum2_right": [wsum2s[R][i] + wsum2s[E][i] for i in range(k * k)],
        "explored_mass": mass,
        "states_explored": states_explored,
    }


def grecov_mass_bfs(p, x_obs, eps=1e-3, tie_margin=1e-8):
    """
    BFS in decreasing multinomial probability order.

    Computes pi_> = P({y : P(y|p) > P(x_obs|p)}) with a tie margin
    to avoid discontinuities at exact ties.

    Parameters
    ----------
    p : list of float
        Probability vector.
    x_obs : list of int
        Observed count vector. n is derived as sum(x_obs).
    eps : float
        Stopping tolerance on unexplored mass.
    tie_margin : float
        Log-probability margin for near-tie exclusion. A state y is
        counted as strictly more probable only if
        log P(y|p) > log P(x_obs|p) + tie_margin.

    Returns
    -------
    dict with keys:
        explored_mass   — P({x : log P(x|p) > log P(x_obs|p) + tie_margin})
        states_explored — number of states popped from the heap
    """
    x_obs = tuple(x_obs)
    n = sum(x_obs)
    assert n > 0, "n must be positive"

    k = len(p)

    MIN_P = 1e-300
    p_stable = [max(pi, MIN_P) for pi in p]
    p_sum = sum(p_stable)
    p_stable = [pi / p_sum for pi in p_stable]

    log_p = [math.log(pi) for pi in p_stable]

    # Precompute log-factorials
    log_fact = [0.0] * (n + 1)
    for i in range(1, n + 1):
        log_fact[i] = log_fact[i - 1] + math.log(i)

    def log_prob(counts):
        val = log_fact[n]
        for ci, lp in zip(counts, log_p):
            val -= log_fact[ci]
            val += ci * lp
        return val

    log_p_obs = log_prob(x_obs)
    threshold = log_p_obs + tie_margin

    # Start from the approximate mode
    start = tuple(_start_counts(p_stable, n))
    heap = [(-log_prob(start), start)]  # min-heap with negated keys = max-heap
    visited = {start}

    mass = 0.0
    states_explored = 0

    while heap:
        neg_lp, counts = heapq.heappop(heap)
        log_p_state = -neg_lp
        states_explored += 1

        # BFS is in decreasing order; once below threshold, no more states contribute
        if log_p_state <= threshold:
            break

        mass += math.exp(log_p_state)

        if 1.0 - eps <= mass:
            break

        # Generate neighbours: transfer one unit from j to i
        for j in range(k):
            if counts[j] == 0:
                continue
            for i in range(k):
                if i == j:
                    continue
                neighbor = list(counts)
                neighbor[i] += 1
                neighbor[j] -= 1
                neighbor = tuple(neighbor)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                log_p_neighbor = (
                    log_p_state
                    + math.log(counts[j])
                    - math.log(counts[i] + 1)
                    + log_p[i]
                    - log_p[j]
                )
                heapq.heappush(heap, (-log_p_neighbor, neighbor))

    return {
        "explored_mass": mass,
        "states_explored": states_explored,
    }
