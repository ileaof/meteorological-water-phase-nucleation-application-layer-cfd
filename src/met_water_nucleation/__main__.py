"""Enable ``python -m met_water_nucleation``."""
import sys

from .cli import main

sys.exit(main())