"""
submit.py
=========
Root entry point redirecting to mtah2027_possession/submit.py.
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "mtah2027_possession"))

from mtah2027_possession.submit import main

if __name__ == "__main__":
    main()

