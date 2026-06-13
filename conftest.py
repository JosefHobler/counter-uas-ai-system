"""Pytest path setup.

Puts the repo root (for the `common` package) and `edge-rpi5` (for `config`,
`tracker`, `drone_detector`) on sys.path so tests can import them directly
without an installed package.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, os.path.join(ROOT, "edge-rpi5")):
    if path not in sys.path:
        sys.path.insert(0, path)
