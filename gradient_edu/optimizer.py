"""Core optimization routines for educational use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class GradientDescentResult:
    """Result returned by gradient descent."""

    point: np.ndarray
    value: float
    steps: int
    converged: bool
    path: List[np.ndarray]


def _as_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("start must be a one-dimensional sequence of numbers")
    if vector.size == 0:
        raise ValueError("start must contain at least one value")
    if not np.all(np.isfinite(vector)):
        raise ValueError("start must contain only finite values")
    return vector


def finite_difference_gradient(
    objective: Objective,
    point: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Approximate the gradient at a point using central differences."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    x = _as_vector(point)
    gradient = np.zeros_like(x)

    for index in range(x.size):
        offset = np.zeros_like(x)
        offset[index] = epsilon
        forward = float(objective(x + offset))
        backward = float(objective(x - offset))
        gradient[index] = (forward - backward) / (2.0 * epsilon)

    if not np.all(np.isfinite(gradient)):
        raise ValueError("gradient contains non-finite values")

    return gradient


def gradient_descent(
    objective: Objective,
    start: Sequence[float],
    *,
    gradient: Gradient | None = None,
    learning_rate: float = 0.1,
    max_steps: int = 1000,
    tolerance: float = 1e-8,
    epsilon: float = 1e-6,
) -> GradientDescentResult:
    """Minimize an objective function with basic gradient descent."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    x = _as_vector(start)
    path = [x.copy()]
    converged = False

    for step in range(1, max_steps + 1):
        grad = gradient(x) if gradient is not None else finite_difference_gradient(
            objective,
            x,
            epsilon=epsilon,
        )
        grad = np.asarray(grad, dtype=float)

        if grad.shape != x.shape:
            raise ValueError("gradient must have the same shape as start")
        if not np.all(np.isfinite(grad)):
            raise ValueError("gradient must contain only finite values")

        next_x = x - learning_rate * grad
        if not np.all(np.isfinite(next_x)):
            raise ValueError("optimization produced non-finite values")

        path.append(next_x.copy())

        if np.linalg.norm(next_x - x) <= tolerance:
            x = next_x
            converged = True
            break

        x = next_x
    else:
        step = max_steps

    value = float(objective(x))
    if not np.isfinite(value):
        raise ValueError("objective returned a non-finite value")

    return GradientDescentResult(
        point=x,
        value=value,
        steps=step,
        converged=converged,
        path=path,
    )
