"""Compatibility entry point for the original Gradient script.

The old version accepted arbitrary Python code from input and executed it with
``exec``/``eval``. That behavior is intentionally removed. Use the packaged
examples or import ``gradient_edu`` directly instead.
"""

from gradient_edu.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
