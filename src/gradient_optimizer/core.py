"""Core optimization functions for educational use.

The public API accepts Python callables instead of evaluating user-provided
strings. This keeps examples simple while avoiding arbitrary code execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class OptimizationResult:
    """Result returned by gradient descent or gradient ascent.

    Attributes:
        point: Final point reached by the optimizer.
        value: Objective value at the final point.
        iterations: Number of update steps performed.
        converged: Whether the optimizer stopped because the step was small.
        path: Copy of each point visited during optimization.
    """

    point: np.ndarray
    value: float
    iterations: int
    converged: bool
    path: List[np.ndarray]


def _validate_callable(name: str, value: object) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be a callable that accepts a NumPy array")


def _as_vector(name: str, values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of numbers")
    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def numerical_gradient(
    objective: Objective,
    point: Sequence[float],
    *,
    step_size: float = 1e-6,
) -> np.ndarray:
    """Approximate a gradient using central finite differences.

    Args:
        objective: Function that maps a NumPy vector to a numeric value.
        point: Point where the gradient should be estimated.
        step_size: Small positive distance used for finite differences.

    Returns:
        A NumPy vector with the same shape as ``point``.
    """

    _validate_callable("objective", objective)
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    x = _as_vector("point", point)
    gradient = np.zeros_like(x)

    for index in range(x.size):
        offset = np.zeros_like(x)
        offset[index] = step_size
        forward = float(objective(x + offset))
        backward = float(objective(x - offset))
        gradient[index] = (forward - backward) / (2.0 * step_size)

    if not np.all(np.isfinite(gradient)):
        raise ValueError("objective produced a non-finite numerical gradient")

    return gradient


def gradient_descent(
    objective: Objective,
    start: Sequence[float],
    *,
    gradient: Gradient | None = None,
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    step_size: float = 1e-6,
) -> OptimizationResult:
    """Minimize an objective function with basic gradient descent."""

    return _optimize(
        objective,
        start,
        gradient=gradient,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
        step_size=step_size,
        direction=-1.0,
    )


def gradient_ascent(
    objective: Objective,
    start: Sequence[float],
    *,
    gradient: Gradient | None = None,
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    step_size: float = 1e-6,
) -> OptimizationResult:
    """Maximize an objective function with basic gradient ascent."""

    return _optimize(
        objective,
        start,
        gradient=gradient,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
        step_size=step_size,
        direction=1.0,
    )


def _optimize(
    objective: Objective,
    start: Sequence[float],
    *,
    gradient: Gradient | None,
    learning_rate: float,
    max_iterations: int,
    tolerance: float,
    step_size: float,
    direction: float,
) -> OptimizationResult:
    _validate_callable("objective", objective)
    if gradient is not None:
        _validate_callable("gradient", gradient)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    x = _as_vector("start", start)
    path = [x.copy()]
    converged = False

    for iteration in range(1, max_iterations + 1):
        grad = gradient(x) if gradient is not None else numerical_gradient(
            objective,
            x,
            step_size=step_size,
        )
        grad = np.asarray(grad, dtype=float)

        if grad.shape != x.shape:
            raise ValueError("gradient must return the same shape as start")
        if not np.all(np.isfinite(grad)):
            raise ValueError("gradient must return only finite values")

        next_x = x + direction * learning_rate * grad
        if not np.all(np.isfinite(next_x)):
            raise ValueError("optimization produced non-finite values")

        path.append(next_x.copy())

        if np.linalg.norm(next_x - x) <= tolerance:
            x = next_x
            converged = True
            break

        x = next_x
    else:
        iteration = max_iterations

    value = float(objective(x))
    if not np.isfinite(value):
        raise ValueError("objective must return a finite numeric value")

    return OptimizationResult(
        point=x,
        value=value,
        iterations=iteration,
        converged=converged,
        path=path,
    )
