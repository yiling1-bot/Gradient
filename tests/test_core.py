import numpy as np
import pytest

from gradient_optimizer import gradient_ascent, gradient_descent, numerical_gradient
from gradient_optimizer.examples import (
    quadratic_1d,
    quadratic_1d_gradient,
    quadratic_2d,
    quadratic_2d_gradient,
)


def test_numerical_gradient_on_simple_function():
    gradient = numerical_gradient(lambda x: float((x[0] - 2.0) ** 2), [0.0])

    assert np.allclose(gradient, np.array([-4.0]), atol=1e-5)


def test_gradient_descent_on_1d_quadratic():
    result = gradient_descent(
        quadratic_1d,
        [0.0],
        gradient=quadratic_1d_gradient,
        learning_rate=0.1,
        max_iterations=200,
    )

    assert result.converged
    assert np.allclose(result.point, np.array([2.0]), atol=1e-4)
    assert result.value < 1e-8


def test_gradient_descent_on_2d_quadratic():
    result = gradient_descent(
        quadratic_2d,
        [0.0, 0.0],
        gradient=quadratic_2d_gradient,
        learning_rate=0.1,
        max_iterations=200,
    )

    assert result.converged
    assert np.allclose(result.point, np.array([1.0, -3.0]), atol=1e-4)
    assert result.value < 1e-8


def test_gradient_ascent_on_negative_quadratic():
    def objective(x):
        return float(-((x[0] - 2.0) ** 2))

    def gradient(x):
        return np.array([-2.0 * (x[0] - 2.0)])

    result = gradient_ascent(
        objective,
        [0.0],
        gradient=gradient,
        learning_rate=0.1,
        max_iterations=200,
    )

    assert result.converged
    assert np.allclose(result.point, np.array([2.0]), atol=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"learning_rate": -0.1},
        {"max_iterations": 0},
        {"max_iterations": -1},
    ],
)
def test_gradient_descent_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        gradient_descent(quadratic_1d, [0.0], **kwargs)


def test_gradient_descent_rejects_non_callable_objective():
    with pytest.raises(TypeError, match="objective must be a callable"):
        gradient_descent("not a function", [0.0])


def test_gradient_descent_rejects_non_callable_gradient():
    with pytest.raises(TypeError, match="gradient must be a callable"):
        gradient_descent(quadratic_1d, [0.0], gradient="not a function")


def test_gradient_shape_must_match_start_shape():
    def bad_gradient(x):
        return np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        gradient_descent(quadratic_1d, [0.0], gradient=bad_gradient)
