// C++ implementation of the GreCov BFS algorithm
//
// Optimised best-first enumeration of multinomial count vectors in decreasing
// probability order.  Key optimisations:
// - Pack d-dimensional count vectors into uint64_t words (8-bit when n<=255,
//   16-bit otherwise) for cheap hashing & comparison.
// - Open-addressing hash set (FlatHashSet) for cache-friendly lookups.
// - Template on D (dimension) and packing scheme so inner loops are fully
//   unrolled at compile time.
// - Precomputed log(i) lookup table to avoid repeated std::log() calls.
// - Cached log tables and reusable buffers across calls with same n.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

// ─── MSVC compatibility ─────────────────────────────────────────────

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#define UNREACHABLE() __assume(false)
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#else
#define NOINLINE __attribute__((noinline))
#define UNREACHABLE() __builtin_unreachable()
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// ─── Shared hash finalizer ──────────────────────────────────────────

inline size_t murmur_mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<size_t>(x);
}

// ─── Packing schemes ────────────────────────────────────────────────

// 8-bit per dimension, single uint64_t (d 5-8, n<=255)
template <int D> struct Pack8 {
  static_assert(D >= 5 && D <= 8);
  using State = uint64_t;
  static constexpr State EMPTY = ~uint64_t(0);

  static inline State pack(const int *c) {
    uint64_t s = 0;
    for (int i = 0; i < D; ++i)
      s |= uint64_t(c[i]) << ((7 - i) * 8);
    return s;
  }
  static inline void unpack(State s, int *c) {
    for (int i = 0; i < D; ++i)
      c[i] = int((s >> ((7 - i) * 8)) & 0xFF);
  }
  static inline State neighbor(State s, int i, int j) {
    static constexpr uint64_t D8[8] = {1ULL << 56, 1ULL << 48, 1ULL << 40,
                                       1ULL << 32, 1ULL << 24, 1ULL << 16,
                                       1ULL << 8,  1ULL};
    return s + D8[i] - D8[j];
  }
  static inline size_t hash(State x) { return murmur_mix64(x); }
};

// 16-bit per dimension, single uint64_t (d<=4, any n)
template <int D> struct Pack16 {
  static_assert(D <= 4);
  using State = uint64_t;
  static constexpr State EMPTY = ~uint64_t(0);

  static inline State pack(const int *c) {
    uint64_t s = 0;
    for (int i = 0; i < D; ++i)
      s |= uint64_t(c[i]) << ((3 - i) * 16);
    return s;
  }
  static inline void unpack(State s, int *c) {
    for (int i = 0; i < D; ++i)
      c[i] = int((s >> ((3 - i) * 16)) & 0xFFFF);
  }
  static inline State neighbor(State s, int i, int j) {
    static constexpr uint64_t D16[4] = {1ULL << 48, 1ULL << 32, 1ULL << 16,
                                        1ULL};
    return s + D16[i] - D16[j];
  }
  static inline size_t hash(State x) { return murmur_mix64(x); }
};

// 128-bit unsigned integer for packing d 5-8 with 16-bit slots when n>255.
// Uses __uint128_t on GCC/Clang, emulated struct on MSVC.
// Layout: big-endian, dim 0 at bits 112-127, dim 7 at bits 0-15.
// hi holds dims 0-3 (bits 64-127), lo holds dims 4-7 (bits 0-63).

#ifdef _MSC_VER
struct UInt128 {
  uint64_t hi, lo;

  bool operator==(const UInt128 &o) const { return hi == o.hi && lo == o.lo; }
  bool operator!=(const UInt128 &o) const { return !(*this == o); }
  bool operator>(const UInt128 &o) const {
    return hi > o.hi || (hi == o.hi && lo > o.lo);
  }
};

static inline UInt128 uint128_from_shift(int bit) {
  if (bit >= 64)
    return {uint64_t(1) << (bit - 64), 0};
  else
    return {0, uint64_t(1) << bit};
}

static inline UInt128 uint128_add(UInt128 a, UInt128 b) {
  uint64_t lo = a.lo + b.lo;
  uint64_t hi = a.hi + b.hi + (lo < a.lo ? 1 : 0);
  return {hi, lo};
}

static inline UInt128 uint128_sub(UInt128 a, UInt128 b) {
  uint64_t lo = a.lo - b.lo;
  uint64_t hi = a.hi - b.hi - (a.lo < b.lo ? 1 : 0);
  return {hi, lo};
}
#endif

template <int D> struct Pack16x2 {
  static_assert(D >= 5 && D <= 8);

#ifdef _MSC_VER
  using State = UInt128;
  static constexpr State EMPTY = {~uint64_t(0), ~uint64_t(0)};

  static inline State pack(const int *c) {
    State s = {0, 0};
    for (int i = 0; i < D; ++i) {
      int bit = (7 - i) * 16;
      if (bit >= 64)
        s.hi |= uint64_t(c[i]) << (bit - 64);
      else
        s.lo |= uint64_t(c[i]) << bit;
    }
    return s;
  }
  static inline void unpack(State s, int *c) {
    for (int i = 0; i < D; ++i) {
      int bit = (7 - i) * 16;
      if (bit >= 64)
        c[i] = static_cast<int>((s.hi >> (bit - 64)) & 0xFFFF);
      else
        c[i] = static_cast<int>((s.lo >> bit) & 0xFFFF);
    }
  }
  static inline State neighbor(State s, int i, int j) {
    return uint128_sub(uint128_add(s, uint128_from_shift((7 - i) * 16)),
                       uint128_from_shift((7 - j) * 16));
  }
  static inline size_t hash(State x) {
    return murmur_mix64(x.lo ^ (x.hi * 0x9e3779b97f4a7c15ULL));
  }
#else
  using State = __uint128_t;
  static constexpr State EMPTY = ~static_cast<State>(0);

  static inline State pack(const int *c) {
    State s = 0;
    for (int i = 0; i < D; ++i)
      s |= static_cast<State>(c[i]) << ((7 - i) * 16);
    return s;
  }
  static inline void unpack(State s, int *c) {
    for (int i = 0; i < D; ++i)
      c[i] = static_cast<int>((s >> ((7 - i) * 16)) & 0xFFFF);
  }
  static inline State neighbor(State s, int i, int j) {
    static constexpr State D16[8] = {
        static_cast<State>(1) << 112, static_cast<State>(1) << 96,
        static_cast<State>(1) << 80,  static_cast<State>(1) << 64,
        static_cast<State>(1) << 48,  static_cast<State>(1) << 32,
        static_cast<State>(1) << 16,  static_cast<State>(1) << 0,
    };
    return s + D16[i] - D16[j];
  }
  static inline size_t hash(State x) {
    uint64_t lo = static_cast<uint64_t>(x);
    uint64_t hi = static_cast<uint64_t>(x >> 64);
    return murmur_mix64(lo ^ (hi * 0x9e3779b97f4a7c15ULL));
  }
#endif
};

// ─── Open-addressing hash set (templated on packing) ────────────────

template <typename Pack> class FlatHashSet {
  using State = typename Pack::State;

  std::vector<State> slots_;
  size_t mask_;
  size_t size_;

public:
  explicit FlatHashSet(size_t initial_cap = 1 << 16)
      : slots_(initial_cap, Pack::EMPTY), mask_(initial_cap - 1), size_(0) {}

  void clear() {
    std::memset(slots_.data(), 0xFF, slots_.size() * sizeof(State));
    size_ = 0;
  }

  void clear_and_shrink(size_t max_cap = 1 << 18) {
    if (slots_.size() > max_cap) {
      slots_.assign(max_cap, Pack::EMPTY);
      mask_ = max_cap - 1;
      size_ = 0;
    } else {
      clear();
    }
  }

  bool insert(State key) {
    if (UNLIKELY(size_ * 4 >= slots_.size() * 3))
      grow();
    size_t idx = Pack::hash(key) & mask_;
    while (true) {
      const State &slot = slots_[idx];
      if (LIKELY(slot == Pack::EMPTY)) {
        slots_[idx] = key;
        ++size_;
        return true;
      }
      if (slot == key)
        return false;
      idx = (idx + 1) & mask_;
    }
  }

private:
  void grow() {
    size_t new_cap = slots_.size() * 2;
    std::vector<State> old_slots(new_cap, Pack::EMPTY);
    std::swap(old_slots, slots_);
    mask_ = new_cap - 1;
    size_ = 0;
    for (const auto &v : old_slots) {
      if (v != Pack::EMPTY)
        insert(v);
    }
  }
};

// ─── Heap entry & comparator ────────────────────────────────────────

template <typename State> using Entry = std::pair<double, State>;

template <typename State> struct EntryCompare {
  bool operator()(const Entry<State> &a, const Entry<State> &b) const {
    return a.first < b.first || (a.first == b.first && a.second > b.second);
  }
};

// ─── Cached log tables ───────────────────────────────────────────

struct LogTables {
  int n = -1;
  std::vector<double> log_fact;
  std::vector<double> log_int;

  void ensure(int nn) {
    if (nn == n)
      return;
    n = nn;
    log_fact.resize(nn + 1);
    log_fact[0] = 0.0;
    for (int i = 1; i <= nn; ++i)
      log_fact[i] = log_fact[i - 1] + std::log(static_cast<double>(i));
    log_int.resize(nn + 1);
    log_int[0] = 0.0;
    for (int i = 1; i <= nn; ++i)
      log_int[i] = std::log(static_cast<double>(i));
  }
};

static LogTables &cached_tables() {
  static LogTables tables;
  return tables;
}

// ─── Reusable per-packing BFS buffers ──────────────────────────────

template <typename Pack> static FlatHashSet<Pack> &get_visited() {
  static FlatHashSet<Pack> visited;
  return visited;
}

// ─── Mode finding (balanced rounding + greedy hill-climb) ──────────

template <int D, typename Pack>
static typename Pack::State start_counts(const double *p, int n) {
  int x[D];
  int total = 0;
  for (int i = 0; i < D; ++i) {
    x[i] = static_cast<int>(p[i] * n);
    total += x[i];
  }
  // Distribute remainder to dimensions with largest fractional parts
  int remainder = n - total;
  if (remainder > 0) {
    int idx[D];
    for (int i = 0; i < D; ++i)
      idx[i] = i;
    std::sort(idx, idx + D, [&](int a, int b) {
      return (p[a] * n - x[a]) > (p[b] * n - x[b]);
    });
    for (int r = 0; r < remainder; ++r)
      x[idx[r]] += 1;
  }
  // Greedy ascent to the mode: transfer j→i while (x_j/(x_i+1))*(p_i/p_j) > 1
  while (true) {
    double best_ratio = 1.0;
    int best_i = -1, best_j = -1;
    for (int j = 0; j < D; ++j) {
      if (x[j] == 0)
        continue;
      for (int i = 0; i < D; ++i) {
        if (i == j)
          continue;
        double ratio = (static_cast<double>(x[j]) / (x[i] + 1)) * (p[i] / p[j]);
        if (ratio > best_ratio) {
          best_ratio = ratio;
          best_i = i;
          best_j = j;
        }
      }
    }
    if (best_i < 0)
      break;
    x[best_i] += 1;
    x[best_j] -= 1;
  }
  return Pack::pack(x);
}

// ─── BFS result ────────────────────────────────────────────────────

struct BFSResult {
  double prob_left;
  double prob_right;
  std::vector<double> wsum_left;
  std::vector<double> wsum_right;
  std::vector<double> wsum2_left;
  std::vector<double> wsum2_right;
  double explored_mass;
  int64_t states_explored;
};

// ─── Templated tail BFS ────────────────────────────────────────────

template <int D, typename Pack>
NOINLINE static BFSResult grecov_tail_impl(const double *p, const double *v,
                                           double S_obs, int n, double eps) {
  using State = typename Pack::State;
  using E = Entry<State>;
  using Cmp = EntryCompare<State>;

  double log_p[D];
  for (int i = 0; i < D; ++i)
    log_p[i] = std::log(p[i]);

  auto &tables = cached_tables();
  tables.ensure(n);
  const auto &log_fact = tables.log_fact;
  const auto &li = tables.log_int;

  auto log_prob = [&](State state) -> double {
    int c[D];
    Pack::unpack(state, c);
    double val = log_fact[n];
    for (int i = 0; i < D; ++i) {
      val -= log_fact[c[i]];
      val += c[i] * log_p[i];
    }
    return val;
  };

  State start_state = start_counts<D, Pack>(p, n);
  double start_lp = log_prob(start_state);

  auto &visited = get_visited<Pack>();
  visited.clear_and_shrink();

  std::vector<E> heap_storage;
  heap_storage.reserve(1 << 16);
  std::priority_queue<E, std::vector<E>, Cmp> heap(Cmp{},
                                                   std::move(heap_storage));
  heap.push({start_lp, start_state});
  visited.insert(start_state);

  double P_explored = 0.0;
  constexpr int dk = D * D;
  double probs[3] = {};
  double wsums[3][D] = {};
  double wsum2s[3][dk] = {};
  int64_t states_explored = 0;

  auto accumulate = [&](int side, double P_state, const int *c) {
    probs[side] += P_state;
    auto &ws = wsums[side];
    auto &ws2 = wsum2s[side];
    for (int i = 0; i < D; ++i) {
      double pci = P_state * c[i];
      ws[i] += pci;
      ws2[i * D + i] += pci * c[i];
      for (int j = i + 1; j < D; ++j) {
        double val = pci * c[j];
        ws2[i * D + j] += val;
        ws2[j * D + i] += val;
      }
    }
  };

  while (!heap.empty()) {
    auto [logP, state] = heap.top();
    heap.pop();

    ++states_explored;
    double P_state = std::exp(logP);
    P_explored += P_state;

    int c[D];
    Pack::unpack(state, c);

    double s_val = 0.0;
    for (int i = 0; i < D; ++i)
      s_val += c[i] * v[i];

    constexpr int L = 0, R = 1, EQ = 2;
    if (s_val < S_obs)
      accumulate(L, P_state, c);
    else if (s_val > S_obs)
      accumulate(R, P_state, c);
    else
      accumulate(EQ, P_state, c);

    if (1.0 - eps <= P_explored)
      break;

    for (int j = 0; j < D; ++j) {
      if (c[j] == 0)
        continue;
      for (int i = 0; i < D; ++i) {
        if (i == j)
          continue;
        State nb = Pack::neighbor(state, i, j);
        if (!visited.insert(nb))
          continue;
        double logP_n = logP + li[c[j]] - li[c[i] + 1] + log_p[i] - log_p[j];
        heap.push({logP_n, nb});
      }
    }
  }

  constexpr int L = 0, R = 1, EQ = 2;
  BFSResult result;
  result.prob_left = probs[L] + probs[EQ];
  result.prob_right = probs[R] + probs[EQ];
  result.wsum_left.resize(D);
  result.wsum_right.resize(D);
  result.wsum2_left.resize(dk);
  result.wsum2_right.resize(dk);
  for (int i = 0; i < D; ++i) {
    result.wsum_left[i] = wsums[L][i] + wsums[EQ][i];
    result.wsum_right[i] = wsums[R][i] + wsums[EQ][i];
  }
  for (int i = 0; i < dk; ++i) {
    result.wsum2_left[i] = wsum2s[L][i] + wsum2s[EQ][i];
    result.wsum2_right[i] = wsum2s[R][i] + wsum2s[EQ][i];
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

// ─── Templated mass BFS ───────────────────────────────────────────

template <int D, typename Pack>
NOINLINE static MassBFSResult grecov_mass_impl(const double *p,
                                               const int *x_obs, int n,
                                               double eps, double tie_margin) {
  using State = typename Pack::State;
  using E = Entry<State>;
  using Cmp = EntryCompare<State>;

  double log_p[D];
  for (int i = 0; i < D; ++i)
    log_p[i] = std::log(p[i]);

  auto &tables = cached_tables();
  tables.ensure(n);
  const auto &log_fact = tables.log_fact;
  const auto &li = tables.log_int;

  auto log_prob = [&](const int *c) -> double {
    double val = log_fact[n];
    for (int i = 0; i < D; ++i) {
      val -= log_fact[c[i]];
      val += c[i] * log_p[i];
    }
    return val;
  };

  double log_p_obs = log_prob(x_obs);
  double threshold = log_p_obs + tie_margin;

  State start_state = start_counts<D, Pack>(p, n);

  int sc[D];
  Pack::unpack(start_state, sc);
  double start_lp = log_prob(sc);

  auto &visited = get_visited<Pack>();
  visited.clear_and_shrink();

  std::vector<E> heap_storage;
  heap_storage.reserve(1 << 16);
  std::priority_queue<E, std::vector<E>, Cmp> heap(Cmp{},
                                                   std::move(heap_storage));
  heap.push({start_lp, start_state});
  visited.insert(start_state);

  double mass = 0.0;
  int64_t states_explored = 0;

  while (!heap.empty()) {
    auto [logP, state] = heap.top();
    heap.pop();

    ++states_explored;

    if (logP <= threshold)
      break;

    mass += std::exp(logP);

    if (1.0 - eps <= mass)
      break;

    int c[D];
    Pack::unpack(state, c);

    for (int j = 0; j < D; ++j) {
      if (c[j] == 0)
        continue;
      for (int i = 0; i < D; ++i) {
        if (i == j)
          continue;
        State nb = Pack::neighbor(state, i, j);
        if (!visited.insert(nb))
          continue;
        double logP_n = logP + li[c[j]] - li[c[i] + 1] + log_p[i] - log_p[j];
        heap.push({logP_n, nb});
      }
    }
  }

  return {mass, states_explored};
}

// ─── Input validation & probability stabilization ──────────────────

static void validate_inputs(int d, int n, int v_size = -1) {
  if (d < 2 || d > 8)
    throw std::runtime_error("dimension must be between 2 and 8");
  if (v_size >= 0 && v_size != d)
    throw std::runtime_error("all input vectors must have the same length");
  if (n <= 0)
    throw std::runtime_error("n must be positive");
  if (n > 65535)
    throw std::runtime_error("n must be <= 65535");
}

static std::vector<double> stabilize_probs(const std::vector<double> &p_raw) {
  constexpr double MIN_P = 1e-300;
  int d = static_cast<int>(p_raw.size());
  std::vector<double> p(d);
  double p_sum = 0.0;
  for (int i = 0; i < d; ++i) {
    p[i] = std::max(p_raw[i], MIN_P);
    p_sum += p[i];
  }
  for (int i = 0; i < d; ++i)
    p[i] /= p_sum;
  return p;
}

// ─── Dispatch by dimension and packing ─────────────────────────────

static BFSResult grecov_tail_dispatch(const std::vector<double> &p_raw,
                                      const std::vector<double> &v,
                                      double S_obs, int n, double eps) {
  int d = static_cast<int>(p_raw.size());
  validate_inputs(d, n, static_cast<int>(v.size()));
  auto p = stabilize_probs(p_raw);
  const double *pp = p.data();
  const double *vp = v.data();

  // d<=4: use Pack16 (uint64_t, 16-bit slots) for all n
  // d>=5, n<=255: use Pack8 (uint64_t, 8-bit slots)
  // d>=5, n>255: use Pack16x2 (__uint128_t, 16-bit slots)
  if (d <= 4) {
#define DISPATCH_BFS_16(D_VAL)                                                 \
  case D_VAL:                                                                  \
    return grecov_tail_impl<D_VAL, Pack16<D_VAL>>(pp, vp, S_obs, n, eps);
    switch (d) {
      DISPATCH_BFS_16(2)
      DISPATCH_BFS_16(3)
      DISPATCH_BFS_16(4)
    }
#undef DISPATCH_BFS_16
  } else if (n <= 255) {
#define DISPATCH_BFS_8(D_VAL)                                                  \
  case D_VAL:                                                                  \
    return grecov_tail_impl<D_VAL, Pack8<D_VAL>>(pp, vp, S_obs, n, eps);
    switch (d) {
      DISPATCH_BFS_8(5)
      DISPATCH_BFS_8(6)
      DISPATCH_BFS_8(7)
      DISPATCH_BFS_8(8)
    }
#undef DISPATCH_BFS_8
  } else {
#define DISPATCH_BFS_16X2(D_VAL)                                               \
  case D_VAL:                                                                  \
    return grecov_tail_impl<D_VAL, Pack16x2<D_VAL>>(pp, vp, S_obs, n, eps);
    switch (d) {
      DISPATCH_BFS_16X2(5)
      DISPATCH_BFS_16X2(6)
      DISPATCH_BFS_16X2(7)
      DISPATCH_BFS_16X2(8)
    }
#undef DISPATCH_BFS_16X2
  }
  UNREACHABLE();
}

static MassBFSResult grecov_mass_dispatch(const std::vector<double> &p_raw,
                                          const std::vector<int32_t> &x_obs,
                                          double eps, double tie_margin) {
  int d = static_cast<int>(p_raw.size());
  int n = 0;
  for (auto xi : x_obs)
    n += xi;
  validate_inputs(d, n, static_cast<int>(x_obs.size()));
  auto p = stabilize_probs(p_raw);

  std::vector<int> x(x_obs.begin(), x_obs.end());
  const double *pp = p.data();
  const int *xp = x.data();

  if (d <= 4) {
#define DISPATCH_MASS_16(D_VAL)                                                \
  case D_VAL:                                                                  \
    return grecov_mass_impl<D_VAL, Pack16<D_VAL>>(pp, xp, n, eps, tie_margin);
    switch (d) {
      DISPATCH_MASS_16(2)
      DISPATCH_MASS_16(3)
      DISPATCH_MASS_16(4)
    }
#undef DISPATCH_MASS_16
  } else if (n <= 255) {
#define DISPATCH_MASS_8(D_VAL)                                                 \
  case D_VAL:                                                                  \
    return grecov_mass_impl<D_VAL, Pack8<D_VAL>>(pp, xp, n, eps, tie_margin);
    switch (d) {
      DISPATCH_MASS_8(5)
      DISPATCH_MASS_8(6)
      DISPATCH_MASS_8(7)
      DISPATCH_MASS_8(8)
    }
#undef DISPATCH_MASS_8
  } else {
#define DISPATCH_MASS_16X2(D_VAL)                                              \
  case D_VAL:                                                                  \
    return grecov_mass_impl<D_VAL, Pack16x2<D_VAL>>(pp, xp, n, eps, tie_margin);
    switch (d) {
      DISPATCH_MASS_16X2(5)
      DISPATCH_MASS_16X2(6)
      DISPATCH_MASS_16X2(7)
      DISPATCH_MASS_16X2(8)
    }
#undef DISPATCH_MASS_16X2
  }
  UNREACHABLE();
}

// ─── Profile LR statistic ─────────────────────────────────────────
// Compute -2 log(max_{q: v^Tq=mu} L(q) / max_q L(q)) for a single
// count vector. Handles interior and boundary (zero-count) cases.

static double
newton_root(const std::function<std::pair<double, double>(double)> &f_df,
            double x0, double lo, double hi, int maxiter = 60) {
  constexpr double EPS = 1e-12;
  double x = x0;
  for (int i = 0; i < maxiter; ++i) {
    auto [f, df] = f_df(x);
    if (std::abs(df) < EPS)
      break;
    double prev = x;
    x -= f / df;
    if (x <= lo)
      x = (prev + lo) / 2;
    else if (x >= hi)
      x = (prev + hi) / 2;
    if (std::abs(x - prev) < EPS)
      break;
  }
  return x;
}

static double profile_lr(const int *counts, const double *values, int k,
                         double mu) {
  constexpr double EPS = 1e-12;
  double vmin = values[0], vmax = values[0];
  for (int i = 1; i < k; ++i) {
    vmin = std::min(vmin, values[i]);
    vmax = std::max(vmax, values[i]);
  }
  // Split into positive-count and zero-count categories
  double pos_c[8], pos_a[8], zero_v[8];
  int np = 0, nz = 0;
  double n = 0;
  for (int i = 0; i < k; ++i) {
    double ci = counts[i], vi = values[i];
    if (ci > 0) {
      pos_c[np] = ci;
      pos_a[np] = vi - mu;
      n += ci;
      np++;
    } else {
      zero_v[nz++] = vi;
    }
  }
  if (n == 0)
    return INFINITY;

  // mu = sample mean → T = 0. Check before support bounds.
  double mean_err = 0;
  for (int i = 0; i < np; ++i)
    mean_err += pos_c[i] * pos_a[i];
  if (std::abs(mean_err / n) < EPS)
    return 0.0;

  if (mu < vmin || mu > vmax)
    return INFINITY;

  // Feasibility bounds for lambda
  double lo = -1e15, hi = 1e15;
  for (int i = 0; i < np; ++i) {
    if (pos_a[i] > 0)
      lo = std::max(lo, -1.0 / pos_a[i]);
    else if (pos_a[i] < 0)
      hi = std::min(hi, -1.0 / pos_a[i]);
  }

  auto lr_at = [&](double lam) -> double {
    double r = 0;
    for (int i = 0; i < np; ++i) {
      double v = 1.0 + lam * pos_a[i];
      if (v <= 0)
        return INFINITY;
      r += pos_c[i] * std::log(v);
    }
    return 2.0 * r;
  };

  double obs_lo = pos_a[0] + mu, obs_hi = pos_a[0] + mu;
  for (int i = 1; i < np; ++i) {
    obs_lo = std::min(obs_lo, pos_a[i] + mu);
    obs_hi = std::max(obs_hi, pos_a[i] + mu);
  }

  // Interior solver: constrained MLE using only positive-count categories.
  double best = INFINITY;
  if (obs_lo < mu && mu < obs_hi) {
    auto g = [&](double lam) -> std::pair<double, double> {
      double s = 0, ds = 0;
      for (int i = 0; i < np; ++i) {
        double d = 1.0 + lam * pos_a[i];
        double p = pos_c[i] / n;
        s += p * pos_a[i] / d;
        ds -= p * pos_a[i] * pos_a[i] / (d * d);
      }
      return {s, ds};
    };
    best = lr_at(newton_root(g, 0.0, lo, hi));
  }

  // Boundary solver: also try placing mass on zero-count categories.
  // This can give a LOWER T even when mu is interior, because the
  // full-simplex constrained MLE may be better.
  if (nz == 0)
    return best;

  // Try all zero-count category values as targets (both sides of observed
  // support)
  for (int iz = 0; iz < nz; ++iz) {
    double target = zero_v[iz];
    bool ok = (target > obs_hi) || (target < obs_lo);
    if (!ok)
      continue;

    // Solve: (mu - M(lam)) / (1 - S(lam)) = target
    auto h = [&](double lam) -> std::pair<double, double> {
      double S = 0, M = 0, dS = 0, dM = 0;
      for (int i = 0; i < np; ++i) {
        double d = 1.0 + lam * pos_a[i];
        double q = pos_c[i] / (n * d);
        double dq = -pos_c[i] * pos_a[i] / (n * d * d);
        S += q;
        M += q * (pos_a[i] + mu);
        dS += dq;
        dM += dq * (pos_a[i] + mu);
      }
      return {(mu - M) - target * (1 - S), -dM + target * dS};
    };

    // Find starting lam where S < 1
    double sign = (target > mu) ? -1.0 : 1.0;

    double lam0 = sign * 0.1;
    bool found_start = false;
    for (int it = 0; it < 100; ++it) {
      bool all_ok = true;
      double Sv = 0;
      for (int i = 0; i < np; ++i) {
        if (1.0 + lam0 * pos_a[i] <= EPS) {
          all_ok = false;
          break;
        }
        Sv += pos_c[i] / (n * (1.0 + lam0 * pos_a[i]));
      }
      if (all_ok && Sv < 1.0 - EPS) {
        found_start = true;
        break;
      }
      lam0 *= 0.5;
    }
    if (!found_start)
      continue;

    double v = lr_at(newton_root(h, lam0, lo, hi));
    if (!std::isinf(v))
      best = std::min(best, v);
  }
  return best;
}

// ─── Profile LR BFS accumulator ──────────────────────────────────
// For a given (theta, values, z, r_z), run BFS and accumulate
// (prob, wsum, wsum2) over the event A_z = {y : T_z(y) >= t_cutoff}.

struct ProfileBFSResult {
  double prob;
  std::vector<double> wsum;  // k
  std::vector<double> wsum2; // k*k
};

template <int D, typename Pack>
NOINLINE static ProfileBFSResult
profile_bfs_impl(const double *p, const double *values, double z,
                 const int *counts_obs, int n, double eps) {
  // Compute t_cutoff inside C++ so the tie comparison uses the same
  // floating-point path as the per-sample profile_lr.
  double t_cutoff = profile_lr(counts_obs, values, D, z);
  using State = typename Pack::State;
  using E = Entry<State>;
  using Cmp = EntryCompare<State>;

  double log_p[D];
  for (int i = 0; i < D; ++i)
    log_p[i] = std::log(p[i]);

  auto &tables = cached_tables();
  tables.ensure(n);
  const auto &log_fact = tables.log_fact;
  const auto &li = tables.log_int;

  State start_state = start_counts<D, Pack>(p, n);
  int sc[D];
  Pack::unpack(start_state, sc);
  double start_lp = log_fact[n];
  for (int i = 0; i < D; ++i) {
    start_lp -= log_fact[sc[i]];
    start_lp += sc[i] * log_p[i];
  }

  auto &visited = get_visited<Pack>();
  visited.clear_and_shrink();

  std::vector<E> heap_storage;
  heap_storage.reserve(1 << 16);
  std::priority_queue<E, std::vector<E>, Cmp> heap(Cmp{},
                                                   std::move(heap_storage));
  heap.push({start_lp, start_state});
  visited.insert(start_state);

  // Precompute a_i = values_i - z for the profile LR inner loop.
  double a[D];
  for (int i = 0; i < D; ++i)
    a[i] = values[i] - z;

  // LR cache: profile_lr(X, z) depends only on X (z is fixed).
  // IPOPT calls us multiple times at different theta; the BFS at each
  // theta overlaps with previous calls. Caching avoids redundant Newton solves.
  struct StateHash {
    size_t operator()(typename Pack::State s) const { return Pack::hash(s); }
  };
  static thread_local std::unordered_map<typename Pack::State, double,
                                         StateHash>
      lr_map;
  static thread_local double lr_cache_z = -1e30;
  if (z != lr_cache_z) {
    lr_map.clear();
    lr_cache_z = z;
  }

  double mass = 0.0;
  double prob = 0.0;
  double wsum[D] = {};
  double wsum2[D * D] = {};

  while (!heap.empty()) {
    auto [logP, state] = heap.top();
    heap.pop();

    double P_state = std::exp(logP);
    mass += P_state;

    int c[D];
    Pack::unpack(state, c);

    // Check LR cache first, then compute if needed
    double t_x;
    auto it = lr_map.find(state);
    if (it != lr_map.end()) {
      t_x = it->second;
    } else {
      t_x = profile_lr(c, values, D, z);
      lr_map[state] = t_x;
    }
    if (t_x >= t_cutoff) {
      prob += P_state;
      for (int i = 0; i < D; ++i) {
        double pci = P_state * c[i];
        wsum[i] += pci;
        wsum2[i * D + i] += pci * c[i];
        for (int j = i + 1; j < D; ++j) {
          double val = pci * c[j];
          wsum2[i * D + j] += val;
          wsum2[j * D + i] += val;
        }
      }
    }

    if (1.0 - eps <= mass)
      break;

    for (int j = 0; j < D; ++j) {
      if (c[j] == 0)
        continue;
      for (int i = 0; i < D; ++i) {
        if (i == j)
          continue;
        State nb = Pack::neighbor(state, i, j);
        if (!visited.insert(nb))
          continue;
        double logP_n = logP + li[c[j]] - li[c[i] + 1] + log_p[i] - log_p[j];
        heap.push({logP_n, nb});
      }
    }
  }

  ProfileBFSResult result;
  result.prob = prob;
  result.wsum.assign(wsum, wsum + D);
  result.wsum2.assign(wsum2, wsum2 + D * D);
  return result;
}

static ProfileBFSResult profile_bfs_dispatch(const std::vector<double> &p_raw,
                                             const std::vector<double> &values,
                                             double z,
                                             const std::vector<int> &counts_obs,
                                             int n, double eps) {
  int d = static_cast<int>(p_raw.size());
  validate_inputs(d, n, static_cast<int>(values.size()));
  auto p = stabilize_probs(p_raw);
  const double *pp = p.data();
  const double *vp = values.data();
  const int *cp = counts_obs.data();

  if (d <= 4) {
#define DISPATCH_PROF_16(D_VAL)                                                \
  case D_VAL:                                                                  \
    return profile_bfs_impl<D_VAL, Pack16<D_VAL>>(pp, vp, z, cp, n, eps);
    switch (d) { DISPATCH_PROF_16(2) DISPATCH_PROF_16(3) DISPATCH_PROF_16(4) }
#undef DISPATCH_PROF_16
  } else if (n <= 255) {
#define DISPATCH_PROF_8(D_VAL)                                                 \
  case D_VAL:                                                                  \
    return profile_bfs_impl<D_VAL, Pack8<D_VAL>>(pp, vp, z, cp, n, eps);
    switch (d) {
      DISPATCH_PROF_8(5)
      DISPATCH_PROF_8(6) DISPATCH_PROF_8(7) DISPATCH_PROF_8(8)
    }
#undef DISPATCH_PROF_8
  } else {
#define DISPATCH_PROF_16X2(D_VAL)                                              \
  case D_VAL:                                                                  \
    return profile_bfs_impl<D_VAL, Pack16x2<D_VAL>>(pp, vp, z, cp, n, eps);
    switch (d) {
      DISPATCH_PROF_16X2(5)
      DISPATCH_PROF_16X2(6) DISPATCH_PROF_16X2(7) DISPATCH_PROF_16X2(8)
    }
#undef DISPATCH_PROF_16X2
  }
  UNREACHABLE();
}

// ─── nanobind module ───────────────────────────────────────────────

NB_MODULE(_ext, m) {
  m.doc() = "C++ for the GreCov algorithm";

  m.def(
      "grecov_tail",
      [](const std::vector<double> &p, const std::vector<double> &v,
         double S_obs, int n, double eps) -> nb::dict {
        auto res = grecov_tail_dispatch(p, v, S_obs, n, eps);

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
      "grecov_mass",
      [](const std::vector<double> &p, const std::vector<int> &x_obs,
         double eps, double tie_margin) -> nb::dict {
        std::vector<int32_t> x(x_obs.begin(), x_obs.end());
        auto res = grecov_mass_dispatch(p, x, eps, tie_margin);

        nb::dict d;
        d["explored_mass"] = res.explored_mass;
        d["states_explored"] = res.states_explored;
        return d;
      },
      nb::arg("p"), nb::arg("x_obs"), nb::arg("eps") = 1e-3,
      nb::arg("tie_margin") = 1e-8,
      "Run the GreCov mass BFS algorithm.\n\n"
      "Computes pi_> = P({y : P(y|p) > P(x_obs|p)}) with tie margin.");

  m.def(
      "profile_bfs",
      [](const std::vector<double> &p, const std::vector<double> &values,
         double z, const std::vector<int> &counts_obs, int n,
         double eps) -> nb::dict {
        auto res = profile_bfs_dispatch(p, values, z, counts_obs, n, eps);

        nb::dict d;
        d["prob"] = res.prob;
        d["wsum"] = res.wsum;
        d["wsum2"] = res.wsum2;
        return d;
      },
      nb::arg("p"), nb::arg("values"), nb::arg("z"), nb::arg("counts_obs"),
      nb::arg("n"), nb::arg("eps") = 1e-6,
      "Run BFS and accumulate P_theta(A_z) with gradient/Hessian data.\n\n"
      "A_z = {y : T_z(y) >= T_z(counts_obs)} where T_z is the profile LR.");
}
