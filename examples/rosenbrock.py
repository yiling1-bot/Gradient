"""Optimize the Rosenbrock function with gradient descent.

The Rosenbrock function is a common optimization demo. It has a curved valley,
so it is harder than a simple quadratic function. The global minimum is near
(1, 1), where the function value is 0.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gradient_optimizer import gradient_descent
from gradient_optimizer.examples import rosenbrock, rosenbrock_gradient


result = gradient_descent(
    rosenbrock,
    start=[-1.2, 1.0],
    gradient=rosenbrock_gradient,
    learning_rate=0.001,
    max_iterations=20_000,
    tolerance=1e-10,
)

print("Approximate minimum point:", result.point)
print("Function value:", result.value)
print("Iterations:", result.iterations)
