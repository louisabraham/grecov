# grecov

Efficient Neyman construction for multinomial distributions with the Greedy Coverage algorithm.

## Installation

```bash
pip install .
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
