"""Script for profiling the C++ BFS hot path under xctrace."""

from grecov._ext import grecov_tail

p = [0.2, 0.2, 0.2, 0.2, 0.2]
v = [0, 1, 2, 3, 4]
x_obs = [30, 25, 20, 15, 10]
n = 100
s_obs = sum(c * vi for c, vi in zip(x_obs, v))
eps = 5e-5

# Warm up
grecov_tail(p, v, s_obs, n, eps)

# Run enough iterations to get ~10s of C++ time
for _ in range(50):
    grecov_tail(p, v, s_obs, n, eps)
