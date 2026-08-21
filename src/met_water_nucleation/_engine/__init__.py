"""Immutable engine bundle for met_water_nucleation.

This subpackage holds the validated, SHA-256-guarded physics core and the
application/diagnosis engine.  These modules are loaded **read-only** by the
parent package via importlib (by path), never modified, and never refactored
together with production code.  Do not import this subpackage's modules
directly; use the top-level ``met_water_nucleation`` API instead.
"""