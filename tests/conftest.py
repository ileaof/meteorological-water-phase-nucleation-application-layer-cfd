"""Pytest bootstrap for the met_water_nucleation package.

Makes the package importable *without* depending on the current working
directory: the project ``src/`` directory (located relative to this file, not
to CWD) is put on ``sys.path`` so ``import met_water_nucleation`` works whether
or not the package has been ``pip install -e .``'d.

Production code never modifies ``sys.path``; this is a test-only convenience.
"""
import os
import sys

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)