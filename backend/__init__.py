"""Local backend for the time-aware chess review app.

Wraps the validated chess_strength classifier as a service: intake, analysis,
and a small local API. The science lives in src/chess_strength and is reused
here, not reimplemented.
"""
