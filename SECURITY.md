# Security Policy

## Supported Versions

This project is early-stage. Security fixes will target the latest version on
the main branch.

## Reporting a Vulnerability

Please open a GitHub issue with a clear description and a minimal reproduction.

## Input Safety

This project does not use `eval()` for user-provided functions. Accepting raw
Python expressions from users is intentionally out of scope until a restricted
parser is implemented and tested.
