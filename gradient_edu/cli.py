"""Command line interface for built-in educational examples."""

from __future__ import annotations

import argparse

from .functions import (
    quadratic_1d,
    quadratic_1d_gradient,
    quadratic_2d,
    quadratic_2d_gradient,
)
from .optimizer import gradient_descent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run gradient descent examples.")
    parser.add_argument(
        "example",
        choices=["quadratic-1d", "quadratic-2d"],
        help="Built-in example to run.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.example == "quadratic-1d":
        result = gradient_descent(
            quadratic_1d,
            start=[0.0],
            gradient=quadratic_1d_gradient,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
        )
    else:
        result = gradient_descent(
            quadratic_2d,
            start=[0.0, 0.0],
            gradient=quadratic_2d_gradient,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
        )

    print(f"point={result.point.tolist()}")
    print(f"value={result.value:.8f}")
    print(f"steps={result.steps}")
    print(f"converged={result.converged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
