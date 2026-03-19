"""Compare BFS performance between native CPython and Pyodide (node.js/wasm).

Usage:
    python profiling/bench_pyodide.py

Requires:
    - Native: pip install -e .
    - Pyodide: pyodide build  (wheel in dist/)
    - Node.js with pyodide: npm install -g pyodide@0.27.7
"""

import json
import os
import subprocess
import sys
import tempfile
import timeit

# ── Shared benchmark parameters ─────────────────────────────────────

CASES = {
    "p_lower": [
        7.14093467e-01,
        1.36442907e-12,
        1.20061383e-12,
        9.95174051e-03,
        2.75954792e-01,
    ],
    "p_upper": [
        5.20119344e-01,
        1.05923135e-12,
        1.14109740e-02,
        1.34394004e-12,
        4.68469682e-01,
    ],
    "p_mle": [0.3, 0.25, 0.2, 0.15, 0.1],
    "p_uniform": [0.2, 0.2, 0.2, 0.2, 0.2],
}
V = [0, 1, 2, 3, 4]
X_OBS = [30, 25, 20, 15, 10]
EPS = 5e-5
N_VALUES = [20, 50, 100]


def fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:>8.1f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:>8.1f} ms"
    else:
        return f"{seconds:>8.2f}  s"


# ── Native benchmarks ───────────────────────────────────────────────


def bench_native():
    from grecov._ext import grecov_tail, grecov_mass

    results = {}

    for n in N_VALUES:
        s_obs = sum(c * v for c, v in zip(X_OBS, V)) * n / 100
        for name, p in CASES.items():
            res = grecov_tail(p, V, s_obs, n, EPS)
            # warm up then measure
            grecov_tail(p, V, s_obs, n, EPS)
            t = timeit.timeit(lambda p=p: grecov_tail(p, V, s_obs, n, EPS), number=1)
            iters = max(1, int(2.0 / t))
            t = (
                timeit.timeit(
                    lambda p=p: grecov_tail(p, V, s_obs, n, EPS), number=iters
                )
                / iters
            )
            results[f"bfs_n{n}_{name}"] = {
                "time": t,
                "states": res["states_explored"],
            }

    # Mass BFS
    for name, p in CASES.items():
        res = grecov_mass(p, X_OBS, EPS, 1e-8)
        grecov_mass(p, X_OBS, EPS, 1e-8)
        t = timeit.timeit(lambda p=p: grecov_mass(p, X_OBS, EPS, 1e-8), number=1)
        iters = max(1, int(2.0 / t))
        t = (
            timeit.timeit(lambda p=p: grecov_mass(p, X_OBS, EPS, 1e-8), number=iters)
            / iters
        )
        results[f"mass_{name}"] = {
            "time": t,
            "states": res["states_explored"],
        }

    return results


# ── Pyodide benchmarks (via node.js subprocess) ─────────────────────

PYODIDE_BENCH_JS = """\
const {{ loadPyodide }} = require('pyodide');
const path = require('path');

(async () => {{
  const py = await loadPyodide();
  await py.loadPackage(['numpy', 'scipy']);

  const wheelPath = path.resolve({wheel_path!r});
  await py.loadPackage('file://' + wheelPath);

  const results = py.runPython(`
import json, time
from grecov._ext import grecov_tail, grecov_mass

cases = {cases_json}
V = {v_json}
X_OBS = {x_obs_json}
EPS = {eps}
N_VALUES = {n_values_json}

results = {{}}

for n in N_VALUES:
    s_obs = sum(c * v for c, v in zip(X_OBS, V)) * n / 100
    for name, p in cases.items():
        res = grecov_tail(p, V, s_obs, n, EPS)
        # warm up
        grecov_tail(p, V, s_obs, n, EPS)
        # measure
        t0 = time.monotonic()
        t1 = time.monotonic()
        overhead = t1 - t0
        t0 = time.monotonic()
        grecov_tail(p, V, s_obs, n, EPS)
        t1 = time.monotonic()
        elapsed = t1 - t0 - overhead
        iters = max(1, int(2.0 / max(elapsed, 1e-9)))
        t0 = time.monotonic()
        for _ in range(iters):
            grecov_tail(p, V, s_obs, n, EPS)
        t1 = time.monotonic()
        elapsed = (t1 - t0) / iters
        results[f"bfs_n{{n}}_{{name}}"] = {{"time": elapsed, "states": res["states_explored"]}}

for name, p in cases.items():
    res = grecov_mass(p, X_OBS, EPS, 1e-8)
    grecov_mass(p, X_OBS, EPS, 1e-8)
    t0 = time.monotonic()
    grecov_mass(p, X_OBS, EPS, 1e-8)
    t1 = time.monotonic()
    elapsed = t1 - t0
    iters = max(1, int(2.0 / max(elapsed, 1e-9)))
    t0 = time.monotonic()
    for _ in range(iters):
        grecov_mass(p, X_OBS, EPS, 1e-8)
    t1 = time.monotonic()
    elapsed = (t1 - t0) / iters
    results[f"mass_{{name}}"] = {{"time": elapsed, "states": res["states_explored"]}}

json.dumps(results)
  `);

  console.log(results);
}})().catch(e => {{ console.error(e); process.exit(1); }});
"""


def find_wheel():
    dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dist")
    for f in sorted(os.listdir(dist), reverse=True):
        if f.endswith("wasm32.whl"):
            return os.path.join(dist, f)
    raise FileNotFoundError("No pyodide wheel found in dist/. Run: pyodide build")


def bench_pyodide():
    wheel_path = find_wheel()
    js_code = PYODIDE_BENCH_JS.format(
        wheel_path=wheel_path,
        cases_json=json.dumps(CASES),
        v_json=json.dumps(V),
        x_obs_json=json.dumps(X_OBS),
        eps=EPS,
        n_values_json=json.dumps(N_VALUES),
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(js_code)
        js_path = f.name

    try:
        env = os.environ.copy()
        if "NODE_PATH" not in env:
            node_path = subprocess.run(
                ["npm", "root", "-g"], capture_output=True, text=True
            ).stdout.strip()
            if node_path:
                env["NODE_PATH"] = node_path
        result = subprocess.run(
            ["node", js_path],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if result.returncode != 0:
            print("Pyodide stderr:", result.stderr, file=sys.stderr)
            raise RuntimeError("Pyodide benchmark failed")
        # Last line of stdout is the JSON
        output = result.stdout.strip().split("\n")[-1]
        return json.loads(output)
    finally:
        os.unlink(js_path)


# ── Comparison ───────────────────────────────────────────────────────


def print_comparison(native, pyodide):
    print(
        f"\n{'benchmark':<24} {'native':>12} {'pyodide':>12} {'ratio':>8} {'states':>10}"
    )
    print("─" * 68)

    for key in sorted(native.keys()):
        nt = native[key]["time"]
        pt = pyodide[key]["time"]
        states = native[key]["states"]
        ratio = pt / nt if nt > 0 else float("inf")
        print(f"{key:<24} {fmt_time(nt)} {fmt_time(pt)} {ratio:>7.1f}x {states:>10}")

    # Summary for n=100 cases
    native_rates = []
    pyodide_rates = []
    for key in native:
        if "n100" in key:
            s = native[key]["states"]
            if s > 100:
                native_rates.append(s / native[key]["time"] / 1e6)
                pyodide_rates.append(s / pyodide[key]["time"] / 1e6)

    if native_rates:
        print(
            f"\n  n=100 avg throughput:  native {sum(native_rates) / len(native_rates):.2f} M states/s"
            f"  |  pyodide {sum(pyodide_rates) / len(pyodide_rates):.2f} M states/s"
        )


if __name__ == "__main__":
    print("Running native benchmarks...")
    native = bench_native()

    print("Running pyodide benchmarks (node.js)...")
    pyodide = bench_pyodide()

    print_comparison(native, pyodide)
