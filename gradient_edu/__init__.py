"""Educational gradient-based optimization tools."""

from .optimizer import GradientDescentResult, finite_difference_gradient, gradient_descent

__all__ = [
    "GradientDescentResult",
    "finite_difference_gradient",
    "gradient_descent",
]
