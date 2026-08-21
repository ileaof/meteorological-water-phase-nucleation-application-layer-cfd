#!/usr/bin/env python
"""Run the meteorological_flow reference demo (20^3, one-way, 60 s).

Thin wrapper around the package API.  Equivalent CLI::

    python -m meteorological_flow.cli --config configs/cold_dry_vs_warm_moist.yaml \
        --grid-resolution 20 --duration 60 --one-way-coupling \
        --output outputs/flow_reference

This is a demonstration-scale run (NOT operational weather prediction).  The
nucleation kernel is used one-way (diagnostic); see docs/flow_guide.md.
"""
from __future__ import annotations

import os
import sys

# allow ``python examples/run_reference_demo.py`` from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from meteorological_flow import Simulation, from_yaml, apply_overrides


def main() -> int:
    cfg_path = os.path.join(_ROOT, "configs", "cold_dry_vs_warm_moist.yaml")
    cfg = from_yaml(cfg_path)
    cfg = apply_overrides(cfg, grid_resolution=20, duration=60.0,
                          output="outputs/flow_reference", one_way=True)
    sim = Simulation(cfg)
    report = sim.run(progress=lambda t, dur, step:
                     print(f"  step {step:5d}  t={t:7.2f}/{dur:.1f}s"))
    print(f"\nDone. max CFL={report['max_cfl']:.3f}, "
          f"max log10I(liquid)={report['final_stats']['log10I_liq_max']:.2f}")
    print(f"Outputs in {cfg.output.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())