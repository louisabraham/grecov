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
#include <queue>
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

// ─── Balanced rounding (largest-remainder method) ──────────────────

template <int D, typename Pack>
static typename Pack::State start_counts(const double *p, int n) {
  int counts[D];
  double frac[D];
  int total = 0;
  for (int i = 0; i < D; ++i) {
    double x = p[i] * n;
    counts[i] = static_cast<int>(std::floor(x));
    frac[i] = x - counts[i];
    total += counts[i];
  }
  int remainder = n - total;
  int idx[D];
  for (int i = 0; i < D; ++i)
    idx[i] = i;
  std::sort(idx, idx + D, [&frac](int a, int b) { return frac[a] > frac[b]; });
  for (int r = 0; r < remainder; ++r)
    counts[idx[r]] += 1;
  return Pack::pack(counts);
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
NOINLINE static BFSResult grecov_bfs_impl(const double *p, const double *v,
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
NOINLINE static MassBFSResult
grecov_mass_bfs_impl(const double *p, const int *x_obs, int n, double eps,
                     double tie_margin) {
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

static BFSResult grecov_bfs_dispatch(const std::vector<double> &p_raw,
                                     const std::vector<double> &v, double S_obs,
                                     int n, double eps) {
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
    return grecov_bfs_impl<D_VAL, Pack16<D_VAL>>(pp, vp, S_obs, n, eps);
    switch (d) {
      DISPATCH_BFS_16(2)
      DISPATCH_BFS_16(3)
      DISPATCH_BFS_16(4)
    }
#undef DISPATCH_BFS_16
  } else if (n <= 255) {
#define DISPATCH_BFS_8(D_VAL)                                                  \
  case D_VAL:                                                                  \
    return grecov_bfs_impl<D_VAL, Pack8<D_VAL>>(pp, vp, S_obs, n, eps);
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
    return grecov_bfs_impl<D_VAL, Pack16x2<D_VAL>>(pp, vp, S_obs, n, eps);
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

static MassBFSResult grecov_mass_bfs_dispatch(const std::vector<double> &p_raw,
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
    return grecov_mass_bfs_impl<D_VAL, Pack16<D_VAL>>(pp, xp, n, eps,          \
                                                      tie_margin);
    switch (d) {
      DISPATCH_MASS_16(2)
      DISPATCH_MASS_16(3)
      DISPATCH_MASS_16(4)
    }
#undef DISPATCH_MASS_16
  } else if (n <= 255) {
#define DISPATCH_MASS_8(D_VAL)                                                 \
  case D_VAL:                                                                  \
    return grecov_mass_bfs_impl<D_VAL, Pack8<D_VAL>>(pp, xp, n, eps,           \
                                                     tie_margin);
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
    return grecov_mass_bfs_impl<D_VAL, Pack16x2<D_VAL>>(pp, xp, n, eps,        \
                                                        tie_margin);
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

// ─── nanobind module ───────────────────────────────────────────────

NB_MODULE(_ext, m) {
  m.doc() = "C++ for the GreCov algorithm";

  m.def(
      "grecov_bfs",
      [](const std::vector<double> &p, const std::vector<double> &v,
         double S_obs, int n, double eps) -> nb::dict {
        auto res = grecov_bfs_dispatch(p, v, S_obs, n, eps);

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
        auto res = grecov_mass_bfs_dispatch(p, x, eps, tie_margin);

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
