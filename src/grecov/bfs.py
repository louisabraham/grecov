"""
Python implementation of the GreCov BFS algorithm.

Best-first enumeration of multinomial count vectors in decreasing
probability order. Returns two-sided tail probabilities and the
conditional expectations needed for analytic gradients.
"""

import heapq
import math


def _start_counts(p, n):
    """Find the multinomial mode by greedy hill-climbing from balanced rounding.

    Start from the largest-remainder rounding of n*p, then greedily
    transfer one unit from j to i whenever (x_j / (x_i+1)) * (p_i / p_j) > 1,
    picking the move with the largest ratio each step.  Converges in < k steps.
    """
    k = len(p)
    x = [int(pi * n) for pi in p]
    # Distribute remainder to dimensions with largest fractional parts
    remainder = n - sum(x)
    if remainder > 0:
        order = sorted(range(k), key=lambda i: p[i] * n - x[i], reverse=True)
        for r in range(remainder):
            x[order[r]] += 1

    # Greedy ascent to the mode
    while True:
        best_ratio = 1.0
        best_i, best_j = -1, -1
        for j in range(k):
            if x[j] == 0:
                continue
            for i in range(k):
                if i == j:
                    continue
                ratio = (x[j] / (x[i] + 1)) * (p[i] / p[j])
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_i, best_j = i, j
        if best_i < 0:
            break
        x[best_i] += 1
        x[best_j] -= 1

    return x


def _stabilize(p):
    """Clamp and renormalize a probability vector."""
    MIN_P = 1e-300
    p_stable = [max(pi, MIN_P) for pi in p]
    p_sum = sum(p_stable)
    return [pi / p_sum for pi in p_stable]


def _log_multinomial(counts, log_p, log_fact, n):
    """Log-multinomial probability given precomputed tables."""
    val = log_fact[n]
    for ci, lp in zip(counts, log_p):
        val -= log_fact[ci]
        val += ci * lp
    return val


def grecov_iter(p, n, eps=1e-4):
    """
    Yield multinomial count vectors in decreasing probability order.

    Parameters
    ----------
    p : list of float
        Probability vector.
    n : int
        Total count.
    eps : float
        Stopping tolerance on unexplored mass.

    Yields
    ------
    (counts, log_prob, prob) : (tuple[int,...], float, float)
        Count vector, log-probability, and probability, in decreasing order.
    """
    assert n > 0, "n must be positive"

    k = len(p)
    p_stable = _stabilize(p)
    log_p = [math.log(pi) for pi in p_stable]

    log_fact = [0.0] * (n + 1)
    for i in range(1, n + 1):
        log_fact[i] = log_fact[i - 1] + math.log(i)

    start = tuple(_start_counts(p_stable, n))
    start_lp = _log_multinomial(start, log_p, log_fact, n)
    heap = [(-start_lp, start)]
    visited = {start}

    mass = 0.0

    while heap:
        neg_lp, counts = heapq.heappop(heap)
        log_p_state = -neg_lp
        p_state = math.exp(log_p_state)
        mass += p_state

        yield counts, log_p_state, p_state

        if 1.0 - eps <= mass:
            break

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


def grecov_tail(p, v, s_obs, n, eps=1e-4):
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
    k = len(p)

    mass = 0.0
    prob_l = 0.0
    prob_r = 0.0
    prob_e = 0.0
    ws_l = [0.0] * k
    ws_r = [0.0] * k
    ws_e = [0.0] * k
    ws2_l = [0.0] * (k * k)
    ws2_r = [0.0] * (k * k)
    ws2_e = [0.0] * (k * k)
    states_explored = 0

    for counts, _log_p, p_state in grecov_iter(p, n, eps):
        states_explored += 1
        mass += p_state

        # Classify: s < s_obs (left), s > s_obs (right), s == s_obs (both)
        s_val = sum(ci * vi for ci, vi in zip(counts, v))

        if s_val < s_obs:
            prob_l += p_state
            ws = ws_l
            ws2 = ws2_l
        elif s_val > s_obs:
            prob_r += p_state
            ws = ws_r
            ws2 = ws2_r
        else:
            prob_e += p_state
            ws = ws_e
            ws2 = ws2_e

        for i in range(k):
            pci = p_state * counts[i]
            ws[i] += pci
            for j in range(i, k):
                val = pci * counts[j]
                ws2[i * k + j] += val
                if i != j:
                    ws2[j * k + i] += val

    # States with s == s_obs contribute to both tails
    return {
        "prob_left": prob_l + prob_e,
        "prob_right": prob_r + prob_e,
        "wsum_left": [ws_l[i] + ws_e[i] for i in range(k)],
        "wsum_right": [ws_r[i] + ws_e[i] for i in range(k)],
        "wsum2_left": [ws2_l[i] + ws2_e[i] for i in range(k * k)],
        "wsum2_right": [ws2_r[i] + ws2_e[i] for i in range(k * k)],
        "explored_mass": mass,
        "states_explored": states_explored,
    }


def grecov_mass(p, x_obs, eps=1e-3, tie_margin=1e-8):
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

    # Compute threshold using the same stabilized p as the enumerator
    p_stable = _stabilize(p)
    log_p = [math.log(pi) for pi in p_stable]
    log_fact = [0.0] * (n + 1)
    for i in range(1, n + 1):
        log_fact[i] = log_fact[i - 1] + math.log(i)
    threshold = _log_multinomial(x_obs, log_p, log_fact, n) + tie_margin

    mass = 0.0
    states_explored = 0

    for _counts, log_p_state, p_state in grecov_iter(p, n, eps):
        states_explored += 1

        # BFS is in decreasing order; once below threshold, no more states contribute
        if log_p_state <= threshold:
            break

        mass += p_state

    return {
        "explored_mass": mass,
        "states_explored": states_explored,
    }


def grecov_coverage(p, v, n, interval_fn, eps=1e-4):
    """
    Approximate the coverage probability of an interval procedure.

    Computes P_p({x : v^T p ∈ interval_fn(x)}) by summing multinomial
    probabilities over the high-probability region.

    Parameters
    ----------
    p : list of float
        True probability vector.
    v : list of float
        Value vector for each category.
    n : int
        Total count (sample size).
    interval_fn : callable
        Function mapping a count vector (tuple of ints) to an interval,
        either as a (lower, upper) tuple or a dict with "lower" and
        "upper" keys.
    eps : float
        Stopping tolerance on unexplored mass.

    Returns
    -------
    dict with keys:
        coverage        — P_p({x : v^T p ∈ interval_fn(x)})
        explored_mass   — total probability mass visited
        states_explored — number of states popped
    """
    target = sum(pi * vi for pi, vi in zip(p, v))
    coverage = 0.0
    mass = 0.0
    states_explored = 0

    for counts, _log_p, prob in grecov_iter(p, n, eps):
        states_explored += 1
        mass += prob
        interval = interval_fn(counts)
        if isinstance(interval, dict):
            lo, hi = interval["lower"], interval["upper"]
        else:
            lo, hi = interval
        if lo <= target <= hi:
            coverage += prob

    return {
        "coverage": coverage,
        "explored_mass": mass,
        "states_explored": states_explored,
    }
