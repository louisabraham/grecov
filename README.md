# grecov

Efficient Neyman construction for multinomial distributions with the Greedy Coverage algorithm.

## Installation

```bash
pip install .
```

For faster optimization (optional):

```bash
pip install ".[ipopt]"
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
from grecov import confidence_interval

result = confidence_interval(
    counts=[10, 10, 20, 60],
    values=[1, 2, 3, 4],
    alpha=0.05,
)
print(f"95% CI: [{result['lower']:.4f}, {result['upper']:.4f}]")
```

## Pyodide (browser / WebAssembly)

grecov compiles to WebAssembly and runs in the browser via [Pyodide](https://pyodide.org).
The C++ BFS extension compiles unchanged; cyipopt is **not** included (the scipy
`trust-constr` fallback is used automatically).

### Building the Pyodide wheel

```bash
pip install pyodide-build
pyodide xbuildenv install
pyodide build
```

This requires the matching Emscripten version (`pyodide config get emscripten_version`).
Install it via [emsdk](https://github.com/emscripten-core/emsdk) and `source emsdk_env.sh`
before running `pyodide build`. The wheel is written to `dist/`.

### Using in a webpage

```html
<script src="https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js"></script>
<script type="module">
const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy", "scipy"]);

// Load the grecov wheel (adjust path or URL to where you host it)
await pyodide.loadPackage("./grecov-0.1.0-cp312-cp312-pyodide_2024_0_wasm32.whl");

pyodide.runPython(`
from grecov import confidence_interval
result = confidence_interval(
    counts=[10, 10, 20, 60],
    values=[1, 2, 3, 4],
    alpha=0.05,
)
print(f"95% CI: [{result['lower']:.4f}, {result['upper']:.4f}]")
`);
</script>
```

### Performance

Pyodide/wasm performance is on par with native CPython (~1.0x). Run the
comparison benchmark with:

```bash
python profiling/bench_pyodide.py
```
