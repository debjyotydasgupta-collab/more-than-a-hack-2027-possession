"""
src/__init__.py
===============
Exposes mtah2027_possession.src modules at the repository root level.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
PACKAGE_DIR = os.path.join(ROOT, "mtah2027_possession")

if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

