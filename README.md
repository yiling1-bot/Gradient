# Gradient Edu

Gradient Edu is an early-stage educational Python project for learning
gradient-based optimization. It provides small, readable implementations of
gradient descent, finite-difference gradients, examples, and tests.

This repository is intended for students and beginners. The goal is clarity
over performance.

## Project Status

This is an early-stage learning project. The current focus is:

- safer function handling without `eval()`
- a small Python package structure
- runnable examples
- unit tests
- beginner-friendly documentation

## Why No `eval()`?

The original learning script accepted user-entered function strings and used
`eval()` to turn them into Python functions. That is unsafe because arbitrary
input can execute arbitrary Python code.

This version does not evaluate raw user input. The public API accepts Python
callables, and the command line interface runs only built-in examples.

## Installation

From the repository root:

```bash
pip install -e ".[test,plot]"
```

For the minimal package:

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np

from gradient_edu import gradient_descent


def bowl(x: np.ndarray) -> float:
    return float((x[0] - 3.0) ** 2)


result = gradient_descent(
    objective=bowl,
    start=[0.0],
    learning_rate=0.1,
    max_steps=200,
)

print(result.point)
print(result.value)
```

## Command Line

Run a built-in one-dimensional quadratic example:

```bash
gradient-edu quadratic-1d
```

Run a two-dimensional example:

```bash
gradient-edu quadratic-2d
```

## Examples

Examples are stored in `examples/`:

- `quadratic_1d.py`
- `quadratic_2d.py`

## Tests

```bash
pytest
```

## Suggested Small Commits

1. `docs: add license and project metadata`
2. `refactor: introduce gradient_edu package`
3. `security: remove eval-based function loading`
4. `test: add optimizer unit tests`
5. `docs: rewrite README for educational use`
6. `examples: add runnable optimization examples`
7. `feat: add simple command line interface`
8. `chore: prepare v0.1.0 release notes`

## Roadmap

- Add more educational examples.
- Add optional plotting helpers.
- Add a safe expression parser based on an AST allowlist.
- Add type checking and linting.
- Publish a `v0.1.0` GitHub release.

## Contributing

Contributions are welcome. Please keep changes small, readable, and focused on
education.
