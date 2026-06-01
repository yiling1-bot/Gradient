"""Minimize f(x, y) = (x - 1)^2 + (y + 3)^2.

The gradient is [2(x - 1), 2(y + 3)], so the minimum is at (1, -3).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gradient_optimizer import gradient_descent
from gradient_optimizer.examples import quadratic_2d, quadratic_2d_gradient


result = gradient_descent(
    quadratic_2d,
    start=[0.0, 0.0],
    gradient=quadratic_2d_gradient,
    learning_rate=0.1,
    max_iterations=200,
)

print("Minimum point:", result.point)
print("Minimum value:", result.value)
print("Iterations:", result.iterations)
