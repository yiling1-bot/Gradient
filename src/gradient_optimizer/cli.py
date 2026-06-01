"""Small command line interface for built-in examples."""

from __future__ import annotations

import argparse

from .core import gradient_descent
from .examples import (
    quadratic_1d,
    quadratic_1d_gradient,
    quadratic_2d,
    quadratic_2d_gradient,
    rosenbrock,
    rosenbrock_gradient,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""

    parser = argparse.ArgumentParser(description="Run gradient optimization examples.")
    parser.add_argument(
        "example",
        choices=["quadratic-1d", "quadratic-2d", "rosenbrock"],
        help="Built-in example to run.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a built-in example from the command line."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.example == "quadratic-1d":
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.1
        result = gradient_descent(
            quadratic_1d,
            start=[0.0],
            gradient=quadratic_1d_gradient,
            learning_rate=learning_rate,
            max_iterations=args.max_iterations,
        )
    elif args.example == "quadratic-2d":
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.1
        result = gradient_descent(
            quadratic_2d,
            start=[0.0, 0.0],
            gradient=quadratic_2d_gradient,
            learning_rate=learning_rate,
            max_iterations=args.max_iterations,
        )
    else:
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.001
        result = gradient_descent(
            rosenbrock,
            start=[-1.2, 1.0],
            gradient=rosenbrock_gradient,
            learning_rate=learning_rate,
            max_iterations=args.max_iterations,
        )

    print(f"point={result.point.tolist()}")
    print(f"value={result.value:.8f}")
    print(f"iterations={result.iterations}")
    print(f"converged={result.converged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
