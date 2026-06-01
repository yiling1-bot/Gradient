"""Minimize f(x) = (x - 2)^2 with gradient descent.

The derivative is f'(x) = 2(x - 2), so the minimum is at x = 2.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gradient_optimizer import gradient_descent
from gradient_optimizer.examples import quadratic_1d, quadratic_1d_gradient


result = gradient_descent(
    quadratic_1d,
    start=[0.0],
    gradient=quadratic_1d_gradient,
    learning_rate=0.1,
    max_iterations=200,
)

print("Minimum point:", result.point)
print("Minimum value:", result.value)
print("Iterations:", result.iterations)
