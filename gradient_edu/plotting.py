"""Optional plotting helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def plot_1d_path(objective, path: Sequence[np.ndarray]):
    """Plot a one-dimensional optimization path.

    Matplotlib is imported lazily so the core package only depends on NumPy.
    """

    import matplotlib.pyplot as plt

    points = np.array([point[0] for point in path], dtype=float)
    left = float(points.min() - 1.0)
    right = float(points.max() + 1.0)
    xs = np.linspace(left, right, 200)
    ys = [objective(np.array([x])) for x in xs]

    figure, axis = plt.subplots()
    axis.plot(xs, ys, label="objective")
    axis.scatter(points, [objective(np.array([x])) for x in points], color="red")
    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.legend()
    return figure
