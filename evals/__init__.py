"""Scoring functions for the benchmarks.

Kept out of `mlx_asr/` because none of it ships with the CLI: these are the
metrics used to produce RESULTS.md, and they need the `eval` extra (pykakasi,
rapidfuzz). See docs/BENCHMARKS.md for what each one measures and when it is the
wrong choice.
"""
