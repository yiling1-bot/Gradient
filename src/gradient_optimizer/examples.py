"""Objective functions used by examples and tests."""

from __future__ import annotations

import numpy as np


def quadratic_1d(x: np.ndarray) -> float:
    """Return f(x) = (x - 2)^2, whose minimum is at x = 2."""

    return float((x[0] - 2.0) ** 2)


def quadratic_1d_gradient(x: np.ndarray) -> np.ndarray:
    """Return the gradient of f(x) = (x - 2)^2."""

    return np.array([2.0 * (x[0] - 2.0)])


def quadratic_2d(x: np.ndarray) -> float:
    """Return f(x, y) = (x - 1)^2 + (y + 3)^2."""

    return float((x[0] - 1.0) ** 2 + (x[1] + 3.0) ** 2)


def quadratic_2d_gradient(x: np.ndarray) -> np.ndarray:
    """Return the gradient of f(x, y) = (x - 1)^2 + (y + 3)^2."""

    return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] + 3.0)])


def rosenbrock(x: np.ndarray) -> float:
    """Return the Rosenbrock function in two dimensions."""

    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Return the analytic gradient of the two-dimensional Rosenbrock function."""

    return np.array(
        [
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
            200.0 * (x[1] - x[0] ** 2),
        ]
    )
