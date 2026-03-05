// C++ implementation of the GreCov BFS algorithm
//
// Best-first enumeration of multinomial count vectors in decreasing
// probability order.  Returns two-sided tail probabilities and the
// weighted sums sum_x P(x)*x for each tail, needed for analytic gradients.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

// ─── Hash for state vectors ────────────────────────────────────────

struct StateHash {
  std::size_t operator()(const std::vector<int32_t> &v) const noexcept {
    // FNV-1a
    std::size_t h = 14695981039346656037ULL;
    for (auto x : v) {
      h ^= static_cast<std::size_t>(static_cast<uint32_t>(x));
      h *= 1099511628211ULL;
    }
    return h;
  }
};

// ─── Balanced rounding (largest-remainder method) ──────────────────

static std::vector<int32_t> start_counts(const std::vector<double> &p, int n) {
  int d = static_cast<int>(p.size());
  std::vector<int32_t> counts(d);
  std::vector<double> frac(d);

  int total = 0;
  for (int i = 0; i < d; ++i) {
    double x = p[i] * n;
    counts[i] = static_cast<int32_t>(std::floor(x));
    frac[i] = x - counts[i];
    total += counts[i];
  }

  int remainder = n - total;

  // Indices sorted by descending fractional part
  std::vector<int> idx(d);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&frac](int a, int b) { return frac[a] > frac[b]; });

  for (int r = 0; r < remainder; ++r) {
    counts[idx[r]] += 1;
  }
  return counts;
}

// ─── Precompute log-factorials ─────────────────────────────────────

static std::vector<double> log_factorials(int n) {
  std::vector<double> lf(n + 1, 0.0);
  for (int i = 1; i <= n; ++i) {
    lf[i] = lf[i - 1] + std::log(static_cast<double>(i));
  }
  return lf;
}

// ─── Heap entry comparison ─────────────────────────────────────────
// Max-heap by logP, with count vector as tiebreaker (matches Python's
// min-heap on (-logP, counts_tuple) for deterministic ordering).

using Entry = std::pair<double, std::vector<int32_t>>;

struct EntryCompare {
  bool operator()(const Entry &a, const Entry &b) const {
    if (a.first != b.first)
      return a.first < b.first;
    return a.second > b.second; // smaller counts = higher priority
  }
};

// ─── BFS result ────────────────────────────────────────────────────

struct BFSResult {
  double prob_left;
  double prob_right;
  std::vector<double> wsum_left;
  std::vector<double> wsum_right;
  std::vector<double>
      wsum2_left; // k*k flattened: wsum2[i*k+j] = sum P(x)*x_i*x_j
  std::vector<double> wsum2_right;
  double explored_mass;
  int64_t states_explored;
};

// ─── GreCov BFS ────────────────────────────────────────────────────

static BFSResult grecov_bfs_impl(const std::vector<double> &p_raw,
                                 const std::vector<double> &v, double S_obs,
                                 int n, double eps) {
  int d = static_cast<int>(p_raw.size());

  // Stabilise probabilities
  constexpr double MIN_P = 1e-300;
  std::vector<double> p(d);
  double p_sum = 0.0;
  for (int i = 0; i < d; ++i) {
    p[i] = std::max(p_raw[i], MIN_P);
    p_sum += p[i];
  }
  for (int i = 0; i < d; ++i)
    p[i] /= p_sum;

  std::vector<double> log_p(d);
  for (int i = 0; i < d; ++i)
    log_p[i] = std::log(p[i]);

  auto log_fact = log_factorials(n);

  // Compute log-probability of a state
  auto log_prob = [&](const std::vector<int32_t> &c) -> double {
    double val = log_fact[n];
    for (int i = 0; i < d; ++i) {
      val -= log_fact[c[i]];
      val += c[i] * log_p[i];
    }
    return val;
  };

  // Start from approximate mode
  auto start = start_counts(p, n);
  double start_lp = log_prob(start);

  std::priority_queue<Entry, std::vector<Entry>, EntryCompare> heap;

  std::unordered_set<std::vector<int32_t>, StateHash> visited;
  visited.reserve(1 << 16);

  heap.push({start_lp, start});
  visited.insert(start);

  double P_explored = 0.0;
  int dk = d * d;
  // Per-tail accumulators: 0=left, 1=right, 2=equal
  double probs[3] = {};
  std::vector<double> wsums[3], wsum2s[3];
  for (int t = 0; t < 3; ++t) {
    wsums[t].assign(d, 0.0);
    wsum2s[t].assign(dk, 0.0);
  }
  int64_t states_explored = 0;

  auto accumulate = [&](int side, double P_state,
                        const std::vector<int32_t> &c) {
    probs[side] += P_state;
    auto &ws = wsums[side];
    auto &ws2 = wsum2s[side];
    for (int i = 0; i < d; ++i) {
      double pci = P_state * c[i];
      ws[i] += pci;
      for (int j = i; j < d; ++j) {
        double val = pci * c[j];
        ws2[i * d + j] += val;
        if (i != j)
          ws2[j * d + i] += val;
      }
    }
  };

  while (!heap.empty()) {
    auto [logP, counts] = std::move(const_cast<Entry &>(heap.top()));
    heap.pop();

    ++states_explored;
    double P_state = std::exp(logP);
    P_explored += P_state;

    // Compute S = v^T x
    double s_val = 0.0;
    for (int i = 0; i < d; ++i)
      s_val += counts[i] * v[i];

    constexpr int L = 0, R = 1, E = 2;
    if (s_val < S_obs)
      accumulate(L, P_state, counts);
    else if (s_val > S_obs)
      accumulate(R, P_state, counts);
    else
      accumulate(E, P_state, counts);

    if (1.0 - eps <= P_explored)
      break;

    // Generate neighbours: transfer one unit from j to i
    for (int j = 0; j < d; ++j) {
      if (counts[j] == 0)
        continue;
      for (int i = 0; i < d; ++i) {
        if (i == j)
          continue;

        auto neighbor = counts;
        neighbor[i] += 1;
        neighbor[j] -= 1;

        if (!visited.insert(neighbor).second)
          continue;

        double logP_n = logP + std::log(static_cast<double>(counts[j])) -
                        std::log(static_cast<double>(counts[i] + 1)) +
                        log_p[i] - log_p[j];

        heap.push({logP_n, std::move(neighbor)});
      }
    }
  }

  // Merge equal into both tails
  constexpr int L = 0, R = 1, E = 2;
  BFSResult result;
  result.prob_left = probs[L] + probs[E];
  result.prob_right = probs[R] + probs[E];
  result.wsum_left.resize(d);
  result.wsum_right.resize(d);
  result.wsum2_left.resize(dk);
  result.wsum2_right.resize(dk);
  for (int i = 0; i < d; ++i) {
    result.wsum_left[i] = wsums[L][i] + wsums[E][i];
    result.wsum_right[i] = wsums[R][i] + wsums[E][i];
  }
  for (int i = 0; i < dk; ++i) {
    result.wsum2_left[i] = wsum2s[L][i] + wsum2s[E][i];
    result.wsum2_right[i] = wsum2s[R][i] + wsum2s[E][i];
  }
  result.explored_mass = std::min(P_explored, 1.0);
  result.states_explored = states_explored;
  return result;
}

// ─── Mass BFS result ──────────────────────────────────────────────

struct MassBFSResult {
  double explored_mass;
  int64_t states_explored;
};

// ─── GreCov Mass BFS ──────────────────────────────────────────────

static MassBFSResult grecov_mass_bfs_impl(const std::vector<double> &p_raw,
                                          const std::vector<int32_t> &x_obs,
                                          double eps, double tie_margin) {
  int d = static_cast<int>(p_raw.size());
  int n = 0;
  for (auto xi : x_obs)
    n += xi;

  // Stabilise probabilities
  constexpr double MIN_P = 1e-300;
  std::vector<double> p(d);
  double p_sum = 0.0;
  for (int i = 0; i < d; ++i) {
    p[i] = std::max(p_raw[i], MIN_P);
    p_sum += p[i];
  }
  for (int i = 0; i < d; ++i)
    p[i] /= p_sum;

  std::vector<double> log_p(d);
  for (int i = 0; i < d; ++i)
    log_p[i] = std::log(p[i]);

  auto log_fact = log_factorials(n);

  auto log_prob = [&](const std::vector<int32_t> &c) -> double {
    double val = log_fact[n];
    for (int i = 0; i < d; ++i) {
      val -= log_fact[c[i]];
      val += c[i] * log_p[i];
    }
    return val;
  };

  double log_p_obs = log_prob(x_obs);
  double threshold = log_p_obs + tie_margin;

  // Start from approximate mode
  auto start = start_counts(p, n);
  double start_lp = log_prob(start);

  std::priority_queue<Entry, std::vector<Entry>, EntryCompare> heap;

  std::unordered_set<std::vector<int32_t>, StateHash> visited;
  visited.reserve(1 << 16);

  heap.push({start_lp, start});
  visited.insert(start);

  double mass = 0.0;
  int64_t states_explored = 0;

  while (!heap.empty()) {
    auto [logP, counts] = std::move(const_cast<Entry &>(heap.top()));
    heap.pop();

    ++states_explored;

    // BFS is in decreasing order; once at or below threshold, stop
    if (logP <= threshold)
      break;

    mass += std::exp(logP);

    if (1.0 - eps <= mass)
      break;

    // Generate neighbours: transfer one unit from j to i
    for (int j = 0; j < d; ++j) {
      if (counts[j] == 0)
        continue;
      for (int i = 0; i < d; ++i) {
        if (i == j)
          continue;

        auto neighbor = counts;
        neighbor[i] += 1;
        neighbor[j] -= 1;

        if (!visited.insert(neighbor).second)
          continue;

        double logP_n = logP + std::log(static_cast<double>(counts[j])) -
                        std::log(static_cast<double>(counts[i] + 1)) +
                        log_p[i] - log_p[j];

        heap.push({logP_n, std::move(neighbor)});
      }
    }
  }

  return {mass, states_explored};
}

// ─── nanobind module ───────────────────────────────────────────────

NB_MODULE(_ext, m) {
  m.doc() = "C++ for the GreCov algorithm";

  m.def(
      "grecov_bfs",
      [](const std::vector<double> &p, const std::vector<double> &v,
         double S_obs, int n, double eps) -> nb::dict {
        auto res = grecov_bfs_impl(p, v, S_obs, n, eps);

        nb::dict d;
        d["prob_left"] = res.prob_left;
        d["prob_right"] = res.prob_right;
        d["wsum_left"] = res.wsum_left;
        d["wsum_right"] = res.wsum_right;
        d["wsum2_left"] = res.wsum2_left;
        d["wsum2_right"] = res.wsum2_right;
        d["explored_mass"] = res.explored_mass;
        d["states_explored"] = res.states_explored;
        return d;
      },
      nb::arg("p"), nb::arg("v"), nb::arg("S_obs"), nb::arg("n"),
      nb::arg("eps") = 1e-4, "Run the GreCov equal-tail BFS algorithm.");

  m.def(
      "grecov_mass_bfs",
      [](const std::vector<double> &p, const std::vector<int> &x_obs,
         double eps, double tie_margin) -> nb::dict {
        std::vector<int32_t> x(x_obs.begin(), x_obs.end());
        auto res = grecov_mass_bfs_impl(p, x, eps, tie_margin);

        nb::dict d;
        d["explored_mass"] = res.explored_mass;
        d["states_explored"] = res.states_explored;
        return d;
      },
      nb::arg("p"), nb::arg("x_obs"), nb::arg("eps") = 1e-3,
      nb::arg("tie_margin") = 1e-8,
      "Run the GreCov mass BFS algorithm.\n\n"
      "Computes pi_> = P({y : P(y|p) > P(x_obs|p)}) with tie margin.");
}
