# Contributing

Thank you for improving Gradient Edu.

## Development Setup

```bash
pip install -e ".[test,plot]"
pytest
```

## Guidelines

- Keep examples beginner-friendly.
- Prefer clear names over clever abstractions.
- Add tests for optimizer behavior and input validation.
- Do not add `eval()` or execution of arbitrary user input.
