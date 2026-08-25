"""met_water_nucleation — meteorological water-phase nucleation application layer.

A thin package facade over the validated, immutable physics engine.  The
engine bundle lives under ``_engine/`` and is loaded **read-only** via
``importlib`` (it is SHA-256-guarded and never modified):

  * ``_engine/unified_h2o_nucleation_climate/unified_h2o_nucleation_climate.py``
    — the validated shifted-equilibrium core (Ferreira Eq.39a/39b).  Cite:
    Ferreira, I. L., "Assessment of Thermodynamic Variables Affecting Phase
    Nucleation", Physica B: Condensed Matter 695 (2024) 416494.
  * ``_engine/met_h2o_nucleation.py`` — the application / diagnosis layer
    (free-energy decomposition, precipitation diagnosis, I/O, visualisation).
  * ``_engine/het_contact_angle.py`` — heterogeneous contact-angle models.
  * ``_engine/Nucleation_model_H2O_vapour_{solid,liquid}_Sim_2026*.py`` — the
    two SHA-256-guarded reference implementations.

The engine's public API is re-exported here so users only need::

    import met_water_nucleation as M
    reps = M.MetNucleationRunner(M.MetInput(...)).evaluate_point(...)

See ``docs/`` for the manual, hypotheses, architecture and migration guide.
"""
from __future__ import annotations

import importlib.util
import os
import sys

__version__ = "1.0.0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH = os.path.join(_HERE, "_engine", "met_h2o_nucleation.py")

# Load the engine by path (its own __file__-relative loader then finds the
# core and het_contact_angle siblings, exactly as when run as a script).
_spec = importlib.util.spec_from_file_location(
    "met_water_nucleation._engine_met", _ENGINE_PATH
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"could not load engine at {_ENGINE_PATH}")
engine = importlib.util.module_from_spec(_spec)
sys.modules["met_water_nucleation._engine_met"] = engine
_spec.loader.exec_module(engine)

# Re-export the engine's public API plus the core module object `un`.
for _name in dir(engine):
    if not _name.startswith("__"):
        globals()[_name] = getattr(engine, _name)

# Explicit aliases for clarity and for downstream code that imports them.
un = engine.un              # the validated physics core module
_engine = engine            # the application/diagnosis engine module

__all__ = [n for n in dir(engine) if not n.startswith("_")]