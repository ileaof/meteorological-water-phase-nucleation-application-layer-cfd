"""precip_microphysics -- single-moment bulk microphysics + evidence-based
precipitation diagnostics on top of the validated second-order nucleation
kernel.

This package resolves the limitation that a high nucleation rate never by itself
implies rain or hail.  It adds the physical chain the nucleation layer lacked --
embryo activation, condensation/deposition growth, collision-coalescence,
riming, aggregation, freezing/melting, hail wet/dry growth, and sedimentation --
and an auditable confidence/diagnostic-level model that only *confirms*
precipitation when the growth-and-survival evidence is actually present.

The validated nucleation core (``met_water_nucleation._engine``) is imported
read-only and never modified; it is consumed exactly as the flow solver does.

Framework-agnostic: the same :class:`~precip_microphysics.scheme.BulkMicrophysics`
and diagnostics serve both the standalone :class:`~precip_microphysics.column.ColumnModel`
(externally supplied profiles) and, in Increment 2, the 3D flow coupling.
"""
from __future__ import annotations

from .column import ColumnModel
from .config import MicrophysicsConfig, ProcessSwitches
from .diagnostics import diagnose
from .evidence import CAVEAT, LEVEL_NAMES, Reason, evaluate_category
from .scheme import BulkMicrophysics
from .state import MicrophysicsState

__version__ = "0.1.0"

__all__ = [
    "BulkMicrophysics",
    "CAVEAT",
    "ColumnModel",
    "LEVEL_NAMES",
    "MicrophysicsConfig",
    "MicrophysicsState",
    "ProcessSwitches",
    "Reason",
    "diagnose",
    "evaluate_category",
    "__version__",
]
