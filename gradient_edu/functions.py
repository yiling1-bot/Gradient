"""Small objective functions used by examples and tests."""

from __future__ import annotations

import numpy as np


def quadratic_1d(x: np.ndarray) -> float:
    """A one-dimensional bowl with minimum at x = 3."""

    return float((x[0] - 3.0) ** 2)


def quadratic_1d_gradient(x: np.ndarray) -> np.ndarray:
    """Analytic gradient for quadratic_1d."""

    return np.array([2.0 * (x[0] - 3.0)])


def quadratic_2d(x: np.ndarray) -> float:
    """A two-dimensional bowl with minimum at (2, -1)."""

    return float((x[0] - 2.0) ** 2 + 2.0 * (x[1] + 1.0) ** 2)


def quadratic_2d_gradient(x: np.ndarray) -> np.ndarray:
    """Analytic gradient for quadratic_2d."""

    return np.array([2.0 * (x[0] - 2.0), 4.0 * (x[1] + 1.0)])
