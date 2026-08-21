"""meteorological_flow: a CPU 3D Boussinesq flow application wrapped around the
validated second-order water-phase nucleation engine (met_water_nucleation).

Batch 1 provides the flow solver + one-way (diagnostic) nucleation coupling.
The validated nucleation kernel is imported read-only; no equations are
modified.  See docs/flow_guide.md for the formulation, consequences, and
limitations.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .config import SimulationConfig, apply_overrides, from_dict, from_yaml
from .grid import Grid
from .simulation import Simulation
from .state import FlowState

__all__ = [
    "FlowState",
    "Grid",
    "Simulation",
    "SimulationConfig",
    "__version__",
    "apply_overrides",
    "from_dict",
    "from_yaml",
]