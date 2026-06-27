"""Enable ``python -m meridian ...`` as an alias for the ``meridian`` console script."""

import sys

from meridian.cli import main

if __name__ == "__main__":
    sys.exit(main())
