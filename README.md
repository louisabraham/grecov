# grecov

[![PyPI](https://img.shields.io/pypi/v/grecov)](https://pypi.org/project/grecov/)
[![Tests](https://github.com/louisabraham/grecov/actions/workflows/test.yml/badge.svg)](https://github.com/louisabraham/grecov/actions/workflows/test.yml)
[![Build](https://github.com/louisabraham/grecov/actions/workflows/build.yml/badge.svg)](https://github.com/louisabraham/grecov/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Efficient Neyman construction for multinomial distributions with the Greedy Coverage algorithm.

## Installation

For the IPOPT optimizer (recommended for best results):

```bash
pip install grecov[ipopt]
```

## Development

Install in editable mode with test dependencies:

```bash
pip install -e ".[test]"
```

After modifying C++ code in `src/ext/`, rebuild with:

```bash
pip install -e .
```

Install pre-commit hooks to run ruff and clang-format on every commit:

```bash
pip install pre-commit
pre-commit install
```

## Usage

```python
from grecov import multinomial_ci

result = multinomial_ci(
    counts=[10, 10, 20, 60],
    values=[1, 2, 3, 4],
    alpha=0.05,
)
print(f"95% CI: [{result['lower']:.4f}, {result['upper']:.4f}]")
```

## R (via reticulate)

An R package is available in `r-package/` that wraps grecov using
[reticulate](https://rstudio.github.io/reticulate/).

### Installation

```r
# Install reticulate if needed
install.packages("reticulate")

# Install the R package from the local directory
install.packages("r-package", repos = NULL, type = "source")

# Install the grecov Python package into reticulate's Python environment
grecov::install_grecov()
```

### Usage

```r
library(grecov)

result <- multinomial_ci(
  counts = c(10, 10, 20, 60),
  values = c(1, 2, 3, 4),
  alpha = 0.05
)
cat(sprintf("95%% CI: [%.4f, %.4f]\n", result$lower, result$upper))
```

The low-level BFS function is also available:

```r
bfs <- grecov_bfs(
  p = c(0.1, 0.1, 0.2, 0.6),
  v = c(1, 2, 3, 4),
  s_obs = 330,
  n = 100L
)
cat(sprintf("P(S <= 330) = %.6f\n", bfs$prob_left))
```

### Options

All parameters from the Python API are available:

```r
result <- multinomial_ci(
  counts = c(10, 10, 20, 60),
  values = c(1, 2, 3, 4),
  alpha = 0.10,
  method = "greedy",
  verbose = 1L
)
```

## Pyodide (browser / WebAssembly)

grecov compiles to WebAssembly and runs in the browser via [Pyodide](https://pyodide.org).
The C++ BFS extension compiles unchanged; cyipopt is **not** included (the scipy
`trust-constr` fallback is used automatically).

### Using in a webpage

The pyodide wheel is attached to each [GitHub Release](https://github.com/louisabraham/grecov/releases).
The following snippet automatically fetches the latest one:

```html
<script src="https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js"></script>
<script type="module">
const pyodide = await loadPyodide();
await pyodide.loadPackage(["micropip", "numpy", "scipy"]);

// Find and install the latest pyodide wheel from GitHub Releases
const res = await fetch("https://api.github.com/repos/louisabraham/grecov/releases/latest");
const release = await res.json();
const wheel = release.assets.find(a => a.name.includes("pyodide"));
await pyodide.runPythonAsync(`
    import micropip
    await micropip.install("${wheel.browser_download_url}")
`);

pyodide.runPython(`
from grecov import multinomial_ci
result = multinomial_ci(
    counts=[10, 10, 20, 60],
    values=[1, 2, 3, 4],
    alpha=0.05,
)
print(f"95% CI: [{result['lower']:.4f}, {result['upper']:.4f}]")
`);
</script>
```

### Building the Pyodide wheel locally

```bash
pip install pyodide-build
pyodide xbuildenv install
pyodide build
```

This requires the matching Emscripten version (`pyodide config get emscripten_version`).
Install it via [emsdk](https://github.com/emscripten-core/emsdk) and `source emsdk_env.sh`
before running `pyodide build`. The wheel is written to `dist/`.

### Performance

Pyodide/wasm performance is on par with native CPython (~1.0x). Run the
comparison benchmark with:

```bash
python profiling/bench_pyodide.py
```
