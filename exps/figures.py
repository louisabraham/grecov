# %%
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    get_ipython()  # type: ignore[name-defined]
except NameError:
    plt.switch_backend("Agg")

from grecov.bfs import grecov_tail, grecov_mass
from grecov.solver import multinomial_ci

# ── 1D explored mass plot ────────────────────────────────────────────────────

xs_1d = np.linspace(0.01, 0.99, 200)
for x_obs_1d in [[0, 50], [25, 50]]:
    explored_mass_1d = []
    for x in xs_1d:
        res = grecov_mass([1 - x, x], x_obs_1d)
        explored_mass_1d.append(res["explored_mass"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(xs_1d, explored_mass_1d)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"explored mass")
    ax.set_title(rf"grecov_mass with $x_\mathrm{{obs}} = {x_obs_1d!r}$")
    plt.tight_layout()
    plt.show()

# %%
# ── Shared setup ─────────────────────────────────────────────────────────────

x_obs = [5, 3, 4]
v = [0, 1, 2]
n = sum(x_obs)
s_obs = sum(ci * vi for ci, vi in zip(x_obs, v))
v_arr = np.array(v)
alpha = 0.05
bound = alpha / 2.0
mass_bound = 1 - alpha

ci_mass = multinomial_ci(x_obs, v_arr.tolist(), alpha=alpha, method="greedy")
ci_tail = multinomial_ci(x_obs, v, alpha=alpha, method="equal_tail")

p_lo_mass, p_up_mass = ci_mass["p_lower"], ci_mass["p_upper"]
p_lo_tail, p_up_tail = ci_tail["p_lower"], ci_tail["p_upper"]


def _softmax(t0, t1):
    theta = np.array([t0, t1, 0.0])
    shift = theta - theta.max()
    e = np.exp(shift)
    return e / e.sum()


def _to_theta(p_vec):
    theta = np.log(p_vec)
    return theta[:2] - theta[2]


# ── Compute grids: reduced (simplex) parametrization ─────────────────────────

n_grid = 200
xs = np.linspace(0.01, 0.98, n_grid)
ys = np.linspace(0.01, 0.98, n_grid)
X, Y = np.meshgrid(xs, ys)

Z_mass = np.full_like(X, np.nan)
Z_left = np.full_like(X, np.nan)
Z_right = np.full_like(X, np.nan)

for i in tqdm(range(n_grid), desc="reduced grid"):
    for j in range(n_grid):
        x, y = X[i, j], Y[i, j]
        if 1 - x - y < 1e-3:
            continue
        p = [x, y, 1 - x - y]
        r_mass = grecov_mass(p, x_obs)
        Z_mass[i, j] = r_mass["explored_mass"]
        r_tail = grecov_tail(p, v, s_obs, n)
        Z_left[i, j] = r_tail["prob_left"]
        Z_right[i, j] = r_tail["prob_right"]

# ── Compute grids: logit (softmax) parametrization ───────────────────────────

n_grid_s = 200
ts = np.linspace(-3, 3, n_grid_s)
T0, T1 = np.meshgrid(ts, ts)

Z_mass_s = np.full_like(T0, np.nan)
Z_left_s = np.full_like(T0, np.nan)
Z_right_s = np.full_like(T0, np.nan)

for i in tqdm(range(n_grid_s), desc="logit grid"):
    for j in range(n_grid_s):
        p = _softmax(T0[i, j], T1[i, j])
        pl = p.tolist()
        r_mass = grecov_mass(pl, x_obs)
        Z_mass_s[i, j] = r_mass["explored_mass"]
        r_tail = grecov_tail(pl, v, s_obs, n)
        Z_left_s[i, j] = r_tail["prob_left"]
        Z_right_s[i, j] = r_tail["prob_right"]

# Precompute p components in logit space for objective
P0_s = np.vectorize(lambda t0, t1: _softmax(t0, t1)[0])(T0, T1)
P1_s = np.vectorize(lambda t0, t1: _softmax(t0, t1)[1])(T0, T1)
P2_s = 1 - P0_s - P1_s

# ── Shared plotting helpers ──────────────────────────────────────────────────

MLE_P = np.array([c / n for c in x_obs])
MLE_THETA = _to_theta(MLE_P)

# Marker styles for the 4 extremum points
MASS_LO_KW: dict = dict(
    color="tab:blue", marker="x", markersize=8, markeredgewidth=2, ls="none"
)
MASS_UP_KW: dict = dict(
    color="tab:blue", marker="x", markersize=8, markeredgewidth=2, ls="none"
)
TAIL_LO_KW: dict = dict(
    color="tab:red", marker="x", markersize=8, markeredgewidth=2, ls="none"
)
TAIL_UP_KW: dict = dict(
    color="tab:red", marker="x", markersize=8, markeredgewidth=2, ls="none"
)


def _add_simplex_common(ax):
    ax.plot([0, 1], [1, 0], "k--", lw=1, label=r"simplex boundary")
    ax.set_xlabel(r"$p_0$")
    ax.set_ylabel(r"$p_1$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def _add_theta_common(ax):
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")


# %%
# ── Figure 1: Mass BFS — reduced (simplex) parametrization ──────────────────

Z_obj = v_arr[0] * X + v_arr[1] * Y + v_arr[2] * (1 - X - Y)
Z_obj[X + Y > 1 - 1e-3] = np.nan
Z_obj_mass_feas = np.where(Z_mass <= mass_bound, Z_obj, np.nan)
Z_obj_mass_feas[np.isnan(Z_mass)] = np.nan

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: mass heatmap
cm1 = ax1.pcolormesh(X, Y, Z_mass, shading="auto", cmap="viridis")
ax1.contour(X, Y, Z_mass, levels=[mass_bound], colors="red", linewidths=2)
fig.colorbar(cm1, ax=ax1, label=r"$\pi_>(p)$")
_add_simplex_common(ax1)
ax1.plot([], [], "r-", lw=2, label=rf"$\pi_>(p) = 1 - \alpha = {mass_bound}$")
ax1.plot(*MLE_P[:2], "k*", markersize=12, label=r"MLE")
ax1.set_title(
    r"$\pi_>(p) = P_p(P_p(X) \geq P_p(x_\mathrm{obs}))$"
    f"\n$x_\\mathrm{{obs}}={x_obs}$, $n={n}$"
)
ax1.legend(loc="upper right")

# Right: objective with optimal points
cm2 = ax2.pcolormesh(X, Y, Z_obj_mass_feas, shading="auto", cmap="viridis")
ax2.contour(X, Y, Z_mass, levels=[mass_bound], colors="red", linewidths=2)
fig.colorbar(cm2, ax=ax2, label=r"$v^\top p$")
_add_simplex_common(ax2)
ax2.plot([], [], "r-", lw=2, label=r"constraint boundary")
ax2.plot(*MLE_P[:2], "k*", markersize=12, label=r"MLE")
ax2.plot(*p_lo_mass[:2], **MASS_LO_KW, label=rf"lower $= {ci_mass['lower']:.4f}$")
ax2.plot(*p_up_mass[:2], **MASS_UP_KW, label=rf"upper $= {ci_mass['upper']:.4f}$")
ax2.set_title(
    rf"Objective $v^\top p$ in feasible region ($\pi_>(p) \leq {mass_bound}$)"
    f"\n$v={v}$"
)
ax2.legend(loc="upper right")

fig.suptitle(
    r"Likelihood-ordered (mass) Neyman — reduced param" + "\n"
    rf"$x_\mathrm{{obs}}={x_obs}$, $n={n}$, $\alpha={alpha}$",
    y=1.02,
)
plt.tight_layout()
plt.show()

# %%
# ── Figure 2: Mass BFS — logit (softmax) parametrization ────────────────────

Z_obj_s = v_arr[0] * P0_s + v_arr[1] * P1_s + v_arr[2] * P2_s
Z_obj_mass_s_feas = np.where(Z_mass_s <= mass_bound, Z_obj_s, np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: mass heatmap
cm1 = ax1.pcolormesh(T0, T1, Z_mass_s, shading="auto", cmap="viridis")
ax1.contour(T0, T1, Z_mass_s, levels=[mass_bound], colors="red", linewidths=2)
fig.colorbar(cm1, ax=ax1, label=r"$\pi_>(p)$")
_add_theta_common(ax1)
ax1.plot([], [], "r-", lw=2, label=rf"$\pi_>(p) = {mass_bound}$")
ax1.plot(*MLE_THETA, "k*", markersize=12, label=r"MLE $\theta$")
ax1.set_title(
    r"$\pi_>(p)$, $p = \mathrm{softmax}(\theta_0, \theta_1, 0)$"
    f"\n$x_\\mathrm{{obs}}={x_obs}$, $n={n}$"
)
ax1.legend(loc="upper right")

# Right: objective with optimal points
cm2 = ax2.pcolormesh(T0, T1, Z_obj_mass_s_feas, shading="auto", cmap="viridis")
ax2.contour(T0, T1, Z_mass_s, levels=[mass_bound], colors="red", linewidths=2)
fig.colorbar(cm2, ax=ax2, label=r"$v^\top p$")
_add_theta_common(ax2)
ax2.plot([], [], "r-", lw=2, label=r"constraint boundary")
ax2.plot(*MLE_THETA, "k*", markersize=12, label=r"MLE $\theta$")
theta_lo_mass = _to_theta(p_lo_mass)
theta_up_mass = _to_theta(p_up_mass)
ax2.plot(*theta_lo_mass, **MASS_LO_KW, label=rf"lower $= {ci_mass['lower']:.4f}$")
ax2.plot(*theta_up_mass, **MASS_UP_KW, label=rf"upper $= {ci_mass['upper']:.4f}$")
ax2.set_title(
    rf"Objective $v^\top p$ in feasible region (softmax param)"
    f"\n$v={v}$"
)
ax2.legend(loc="upper right")

fig.suptitle(
    r"Likelihood-ordered (mass) Neyman — logit param" + "\n"
    rf"$x_\mathrm{{obs}}={x_obs}$, $n={n}$, $\alpha={alpha}$",
    y=1.02,
)
plt.tight_layout()
plt.show()

# %%
# ── Figure 3: Equal-tail BFS — reduced (simplex) parametrization ────────────

Z_feasible_tail = np.where((Z_left >= bound) & (Z_right >= bound), 1.0, 0.0)
Z_feasible_tail[np.isnan(Z_left)] = np.nan
Z_obj_tail_feas = np.where(Z_feasible_tail == 1.0, Z_obj, np.nan)
Z_obj_tail_feas[np.isnan(Z_left)] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: prob_left
cm0 = axes[0].pcolormesh(X, Y, Z_left, shading="auto", cmap="viridis")
axes[0].contour(X, Y, Z_left, levels=[bound], colors="red", linewidths=2)
fig.colorbar(cm0, ax=axes[0], label=r"$P_p(S \leq s_\mathrm{obs})$")
axes[0].set_title(rf"$P_p(S \leq s_\mathrm{{obs}})$, $s_\mathrm{{obs}} = {s_obs}$")

# Middle: prob_right
cm1 = axes[1].pcolormesh(X, Y, Z_right, shading="auto", cmap="viridis")
axes[1].contour(X, Y, Z_right, levels=[bound], colors="red", linewidths=2)
fig.colorbar(cm1, ax=axes[1], label=r"$P_p(S \geq s_\mathrm{obs})$")
axes[1].set_title(rf"$P_p(S \geq s_\mathrm{{obs}})$, $s_\mathrm{{obs}} = {s_obs}$")

# Right: objective with optimal points
cm2 = axes[2].pcolormesh(X, Y, Z_obj_tail_feas, shading="auto", cmap="viridis")
axes[2].contour(X, Y, Z_left, levels=[bound], colors="red", linewidths=2)
axes[2].contour(X, Y, Z_right, levels=[bound], colors="red", linewidths=2)
fig.colorbar(cm2, ax=axes[2], label=r"$v^\top p$")
axes[2].set_title(r"Objective $v^\top p$ in $C_\alpha(x_\mathrm{obs})$" + f"\n$v={v}$")

for ax in axes:
    _add_simplex_common(ax)
    ax.plot(*MLE_P[:2], "k*", markersize=10, label=r"MLE")
    ax.plot([], [], "r-", lw=2, label=rf"contour at $\alpha/2 = {bound}$")
    ax.legend(loc="upper right", fontsize=7)

# Only add optimal points to rightmost
axes[2].plot(*p_lo_tail[:2], **TAIL_LO_KW, label=rf"lower $= {ci_tail['lower']:.4f}$")
axes[2].plot(*p_up_tail[:2], **TAIL_UP_KW, label=rf"upper $= {ci_tail['upper']:.4f}$")
axes[2].legend(loc="upper right", fontsize=7)

fig.suptitle(
    r"Equal-tail (Buehler) Neyman — reduced param" + "\n"
    rf"$x_\mathrm{{obs}}={x_obs}$, $n={n}$, $v={v}$, $\alpha={alpha}$",
    y=1.05,
)
plt.tight_layout()
plt.show()

# %%
# ── Figure 4: Equal-tail BFS — logit (softmax) parametrization ──────────────

Z_feasible_tail_s = np.where((Z_left_s >= bound) & (Z_right_s >= bound), 1.0, 0.0)
Z_obj_tail_s_feas = np.where(Z_feasible_tail_s == 1.0, Z_obj_s, np.nan)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: prob_left
cm0 = axes[0].pcolormesh(T0, T1, Z_left_s, shading="auto", cmap="viridis")
axes[0].contour(T0, T1, Z_left_s, levels=[bound], colors="red", linewidths=2)
fig.colorbar(cm0, ax=axes[0], label=r"$P_p(S \leq s_\mathrm{obs})$")
axes[0].set_title(rf"$P_p(S \leq s_\mathrm{{obs}})$, $s_\mathrm{{obs}}={s_obs}$")

# Middle: prob_right
cm1 = axes[1].pcolormesh(T0, T1, Z_right_s, shading="auto", cmap="viridis")
axes[1].contour(T0, T1, Z_right_s, levels=[bound], colors="red", linewidths=2)
fig.colorbar(cm1, ax=axes[1], label=r"$P_p(S \geq s_\mathrm{obs})$")
axes[1].set_title(rf"$P_p(S \geq s_\mathrm{{obs}})$, $s_\mathrm{{obs}}={s_obs}$")

# Right: objective with optimal points
cm2 = axes[2].pcolormesh(T0, T1, Z_obj_tail_s_feas, shading="auto", cmap="viridis")
axes[2].contour(T0, T1, Z_left_s, levels=[bound], colors="red", linewidths=2)
axes[2].contour(T0, T1, Z_right_s, levels=[bound], colors="red", linewidths=2)
fig.colorbar(cm2, ax=axes[2], label=r"$v^\top p$")
axes[2].set_title(r"Objective $v^\top p$ in $C_\alpha(x_\mathrm{obs})$" + f"\n$v={v}$")

theta_lo_tail = _to_theta(p_lo_tail)
theta_up_tail = _to_theta(p_up_tail)

for ax in axes:
    _add_theta_common(ax)
    ax.plot(*MLE_THETA, "k*", markersize=10, label=r"MLE $\theta$")
    ax.plot([], [], "r-", lw=2, label=rf"contour at $\alpha/2 = {bound}$")
    ax.legend(loc="upper right", fontsize=7)

# Only add optimal points to rightmost
axes[2].plot(*theta_lo_tail, **TAIL_LO_KW, label=rf"lower $= {ci_tail['lower']:.4f}$")
axes[2].plot(*theta_up_tail, **TAIL_UP_KW, label=rf"upper $= {ci_tail['upper']:.4f}$")
axes[2].legend(loc="upper right", fontsize=7)

fig.suptitle(
    r"Equal-tail (Buehler) Neyman — logit param" + "\n"
    rf"$x_\mathrm{{obs}}={x_obs}$, $n={n}$, $v={v}$, $\alpha={alpha}$",
    y=1.05,
)
plt.tight_layout()
plt.show()

# %%
# ── Figure 5: Intersection of mass & Buehler regions — reduced param ────────

fig, ax = plt.subplots(1, 1, figsize=(9, 8))
cm = ax.pcolormesh(X, Y, Z_obj, shading="auto", cmap="viridis")
fig.colorbar(cm, ax=ax, label=r"$v^\top p$")
ax.contour(X, Y, Z_mass, levels=[mass_bound], colors="tab:blue", linewidths=2)
ax.contour(
    X, Y, Z_left, levels=[bound], colors="tab:red", linewidths=1.5, linestyles="--"
)
ax.contour(
    X, Y, Z_right, levels=[bound], colors="tab:red", linewidths=1.5, linestyles="--"
)
_add_simplex_common(ax)
ax.plot(*MLE_P[:2], "k*", markersize=12, label=r"MLE")

# Legend entries for boundaries
ax.plot(
    [], [], "-", color="tab:blue", lw=2, label=r"mass boundary ($\pi_> = 1-\alpha$)"
)
ax.plot(
    [], [], "--", color="tab:red", lw=1.5, label=r"Buehler boundary ($L,R = \alpha/2$)"
)

# Iso-score lines through each extremum
s_mass_lo = float(v_arr @ p_lo_mass)
s_mass_up = float(v_arr @ p_up_mass)
s_tail_lo = float(v_arr @ p_lo_tail)
s_tail_up = float(v_arr @ p_up_tail)
ax.contour(
    X, Y, Z_obj, levels=[s_mass_lo], colors="tab:blue", linewidths=1, linestyles=":"
)
ax.contour(
    X, Y, Z_obj, levels=[s_mass_up], colors="tab:blue", linewidths=1, linestyles=":"
)
ax.contour(
    X, Y, Z_obj, levels=[s_tail_lo], colors="tab:red", linewidths=1, linestyles=":"
)
ax.contour(
    X, Y, Z_obj, levels=[s_tail_up], colors="tab:red", linewidths=1, linestyles=":"
)

# 4 extremum points
ax.plot(*p_lo_mass[:2], **MASS_LO_KW, label=rf"mass lower $= {s_mass_lo:.4f}$")
ax.plot(*p_up_mass[:2], **MASS_UP_KW, label=rf"mass upper $= {s_mass_up:.4f}$")
ax.plot(*p_lo_tail[:2], **TAIL_LO_KW, label=rf"Buehler lower $= {s_tail_lo:.4f}$")
ax.plot(*p_up_tail[:2], **TAIL_UP_KW, label=rf"Buehler upper $= {s_tail_up:.4f}$")
ax.plot([], [], ":", color="gray", lw=1, label=r"iso-$v^\top p$ at extrema")

ax.set_title(
    r"Mass & Buehler feasible regions — reduced param"
    f"\n$x_\\mathrm{{obs}}={x_obs}$, $n={n}$, $v={v}$, $\\alpha={alpha}$"
)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# %%
# ── Figure 6: Intersection of mass & Buehler regions — logit param ──────────

fig, ax = plt.subplots(1, 1, figsize=(9, 8))
cm = ax.pcolormesh(T0, T1, Z_obj_s, shading="auto", cmap="viridis")
fig.colorbar(cm, ax=ax, label=r"$v^\top p$")
ax.contour(T0, T1, Z_mass_s, levels=[mass_bound], colors="tab:blue", linewidths=2)
ax.contour(
    T0, T1, Z_left_s, levels=[bound], colors="tab:red", linewidths=1.5, linestyles="--"
)
ax.contour(
    T0, T1, Z_right_s, levels=[bound], colors="tab:red", linewidths=1.5, linestyles="--"
)
_add_theta_common(ax)
ax.plot(*MLE_THETA, "k*", markersize=12, label=r"MLE $\theta$")

# Legend entries for boundaries
ax.plot(
    [], [], "-", color="tab:blue", lw=2, label=r"mass boundary ($\pi_> = 1-\alpha$)"
)
ax.plot(
    [], [], "--", color="tab:red", lw=1.5, label=r"Buehler boundary ($L,R = \alpha/2$)"
)

# Iso-score curves through each extremum
ax.contour(
    T0, T1, Z_obj_s, levels=[s_mass_lo], colors="tab:blue", linewidths=1, linestyles=":"
)
ax.contour(
    T0, T1, Z_obj_s, levels=[s_mass_up], colors="tab:blue", linewidths=1, linestyles=":"
)
ax.contour(
    T0, T1, Z_obj_s, levels=[s_tail_lo], colors="tab:red", linewidths=1, linestyles=":"
)
ax.contour(
    T0, T1, Z_obj_s, levels=[s_tail_up], colors="tab:red", linewidths=1, linestyles=":"
)

# 4 extremum points
ax.plot(*theta_lo_mass, **MASS_LO_KW, label=rf"mass lower $= {s_mass_lo:.4f}$")
ax.plot(*theta_up_mass, **MASS_UP_KW, label=rf"mass upper $= {s_mass_up:.4f}$")
ax.plot(*theta_lo_tail, **TAIL_LO_KW, label=rf"Buehler lower $= {s_tail_lo:.4f}$")
ax.plot(*theta_up_tail, **TAIL_UP_KW, label=rf"Buehler upper $= {s_tail_up:.4f}$")
ax.plot([], [], ":", color="gray", lw=1, label=r"iso-$v^\top p$ at extrema")

ax.set_title(
    r"Mass & Buehler feasible regions — logit param"
    f"\n$x_\\mathrm{{obs}}={x_obs}$, $n={n}$, $v={v}$, $\\alpha={alpha}$"
)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# %%
