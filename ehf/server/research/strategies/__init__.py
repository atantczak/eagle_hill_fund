"""Strategy declarations — thin subclasses on a shared base.

A strategy is ~30-80 lines: an entry signal from the ``signals`` library, a
sizing choice from ``.sizing``, and exit rules from ``.exits``. Importing this
package registers the built-in strategies (Simple Rise/Fall, TSMOM).

The §10 conviction ideas will use a separate lightweight ``Screen`` base.
"""

# Importing the concrete strategies registers them under their RunSpec keys.
from ehf.server.research.strategies import (  # noqa: E402,F401  (import for side effect)
    rise_fall,
    tsmom,
)
from ehf.server.research.strategies.base import Strategy, build_strategy, register

__all__ = ["Strategy", "build_strategy", "register"]
