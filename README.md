# Gradient

Gradient is a beginner-friendly Python project for learning gradient-based
optimization with small, readable examples.

## Why This Project Exists

This project started as a learning script for exploring how gradients can be
used to find minimum and maximum values of functions. It is now organized as a
small educational Python package so students can read the code, run examples,
and write tests without dealing with unnecessary framework complexity.

Gradient is not a production optimization library. It is an educational project
for understanding the basics.

## Features

- Basic gradient descent for minimization.
- Basic gradient ascent for maximization.
- Numerical gradients using central finite differences.
- Callable-based API with no raw `eval()` of user input.
- Beginner-friendly examples for one-dimensional, two-dimensional, and
  Rosenbrock functions.
- Pytest-based test suite.

## Installation

Clone the repository and install it locally:

```bash
git clone https://github.com/yiling1-bot/Gradient.git
cd Gradient
pip install -e ".[test]"
```

For the minimal package only:

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np

from gradient_optimizer import gradient_descent


def objective(x: np.ndarray) -> float:
    return float((x[0] - 2.0) ** 2)


def gradient(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (x[0] - 2.0)])


result = gradient_descent(
    objective,
    start=[0.0],
    gradient=gradient,
    learning_rate=0.1,
    max_iterations=200,
)

print(result.point)
print(result.value)
```

## Examples

Run examples from the repository root:

```bash
python examples/quadratic_1d.py
python examples/quadratic_2d.py
python examples/rosenbrock.py
```

## Testing

Install test dependencies and run pytest:

```bash
pip install -e ".[test]"
pytest
```

The GitHub Actions workflow also runs tests on Python 3.10, 3.11, and 3.12 for
pushes and pull requests.

## Safety Note

This project does not use raw `eval()` on user input. Evaluating arbitrary user
strings can execute arbitrary Python code, including imports, file access, or
other unwanted actions.

The safer design is:

- users pass Python callables directly;
- examples define objective functions normally in code;
- future expression-string support should use a restricted parser, not raw
  `eval()`.

## Project Structure

```text
src/gradient_optimizer/
  __init__.py       Public package exports
  core.py           gradient_descent, gradient_ascent, numerical_gradient
  examples.py       Reusable educational objective functions
  cli.py            Small command line interface for built-in examples
examples/           Runnable beginner examples
tests/              Pytest test suite
.github/            CI workflow and collaboration templates
```

## Roadmap

- Add more beginner examples.
- Add optional plotting examples.
- Add a restricted expression parser for safe math expressions.
- Add more explanation pages for numerical gradients and convergence.
- Prepare small educational releases as the project grows.

## Contributing

Contributions are welcome. This is a small educational project, so good
contributions are usually small and clear:

- improve comments or documentation;
- add a beginner-friendly example;
- add or improve tests;
- simplify code without hiding the math.

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
