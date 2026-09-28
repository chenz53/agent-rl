# Tests

Run `PYTHONPATH=src python3 -m unittest discover -s tests -v` from the repository root.

The standard-library suite covers original artificial scoring, historical stage isolation, oncology patient/time boundaries, corrections, cancellation, malformed clocks, multiline CSV notes, unit-aware examples, and censored outcomes. Real verl integration is checked separately in the remote environment.

Passing these tests establishes software behavior, not clinical validity, real multi-turn tokenization correctness, or improved model capability.
