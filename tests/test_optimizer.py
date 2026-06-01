import numpy as np
import pytest

from gradient_edu import finite_difference_gradient, gradient_descent
from gradient_edu.functions import (
    quadratic_1d,
    quadratic_1d_gradient,
    quadratic_2d,
    quadratic_2d_gradient,
)


def test_finite_difference_gradient_matches_quadratic_gradient():
    gradient = finite_difference_gradient(quadratic_1d, [0.0])

    assert np.allclose(gradient, np.array([-6.0]), atol=1e-5)


def test_gradient_descent_finds_1d_minimum():
    result = gradient_descent(
        quadratic_1d,
        [0.0],
        gradient=quadratic_1d_gradient,
        learning_rate=0.1,
        max_steps=200,
    )

    assert result.converged
    assert np.allclose(result.point, np.array([3.0]), atol=1e-4)
    assert result.value < 1e-8


def test_gradient_descent_finds_2d_minimum():
    result = gradient_descent(
        quadratic_2d,
        [0.0, 0.0],
        gradient=quadratic_2d_gradient,
        learning_rate=0.1,
        max_steps=200,
    )

    assert result.converged
    assert np.allclose(result.point, np.array([2.0, -1.0]), atol=1e-4)
    assert result.value < 1e-8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"learning_rate": -0.1},
        {"max_steps": 0},
        {"tolerance": 0},
    ],
)
def test_gradient_descent_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        gradient_descent(quadratic_1d, [0.0], **kwargs)


def test_gradient_shape_must_match_start_shape():
    def bad_gradient(x):
        return np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        gradient_descent(quadratic_1d, [0.0], gradient=bad_gradient)
