"""Allow ``python -m meteorological_flow``."""
import sys

from .cli import main

sys.exit(main())