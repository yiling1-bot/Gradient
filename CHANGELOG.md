# Changelog

## v0.1.0 - Educational release

This is the first stable educational release of Gradient. It is intended for
learning and classroom-style experimentation, not production optimization.

### Added

- `src/gradient_optimizer` package structure.
- Public API with `gradient_descent`, `gradient_ascent`, and
  `numerical_gradient`.
- Beginner examples for 1D quadratic, 2D quadratic, and Rosenbrock functions.
- Pytest coverage for gradients, optimization behavior, and input validation.
- GitHub Actions workflow for Python 3.10, 3.11, and 3.12.
- English and Simplified Chinese README files.
- MIT License and open-source collaboration templates.

### Changed

- Replaced unsafe user-input execution with a callable-based API.
- Updated documentation to explain why raw `eval()` is avoided.

### Release Notes

Suggested GitHub release title:

```text
Gradient v0.1.0: educational gradient optimization package
```

Suggested release notes:

```text
This v0.1.0 release organizes Gradient as a small educational Python package.
It includes gradient descent, gradient ascent, numerical gradients, runnable
examples, tests, documentation, and CI. The project avoids raw eval() and uses
Python callables for safer beginner-friendly examples.
```
