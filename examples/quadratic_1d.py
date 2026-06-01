"""Run gradient descent on a one-dimensional quadratic function."""

from gradient_edu import gradient_descent
from gradient_edu.functions import quadratic_1d, quadratic_1d_gradient


result = gradient_descent(
    quadratic_1d,
    start=[0.0],
    gradient=quadratic_1d_gradient,
    learning_rate=0.1,
    max_steps=200,
)

print("Minimum point:", result.point)
print("Minimum value:", result.value)
