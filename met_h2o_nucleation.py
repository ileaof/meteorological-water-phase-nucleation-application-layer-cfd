#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
met_h2o_nucleation.py — BACKWARD-COMPATIBILITY SHIM
===================================================

The meteorological water-phase nucleation application layer has been
reorganized into the installable package ``met_water_nucleation``.  The
validated engine now lives at::

    src/met_water_nucleation/_engine/met_h2o_nucleation.py

This shim is kept so legacy commands keep working unchanged::

    python met_h2o_nucleation.py --validate
    python met_h2o_nucleation.py --T 260 --P 70000 --RH 110 --phase-mode both --summary

It emits a DeprecationWarning and delegates to the same engine CLI.  The
recommended replacement is::

    pip install -e .                         # once
    met-water-nucleation --validate           # console script
    python -m met_water_nucleation --validate  # module form

The engine modules are loaded READ-ONLY and never modified; scientific results
are byte-for-byte identical to the pre-reorganization behaviour.
"""
import os
import sys
import warnings

warnings.warn(
    "`met_h2o_nucleation.py` at the repo root is a backward-compatibility "
    "shim. Use the installed package: `pip install -e .` then "
    "`met-water-nucleation` or `python -m met_water_nucleation`.",
    DeprecationWarning,
    stacklevel=2,
)


def _load_main():
    """Resolve and return the engine CLI ``main`` callable."""
    # Preferred path: the installed package.
    try:
        from met_water_nucleation.cli import main as _main  # noqa: WPS433
        return _main
    except Exception:
        pass
    # Fallback: load the engine directly by path (no install required), so the
    # documented `python met_h2o_nucleation.py ...` workflow still works from a
    # fresh clone.  The engine's own __file__-relative loader finds the core.
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(
        here, "src", "met_water_nucleation", "_engine", "met_h2o_nucleation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "met_h2o_nucleation_legacy_engine", engine_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


if __name__ == "__main__":
    sys.exit(_load_main()(sys.argv[1:]))