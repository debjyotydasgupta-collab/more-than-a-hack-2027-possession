"""
mtah2027_possession — Ball Possession Tracking Engine
FC Barcelona Hackathon "More Than A Hack 2027"

Package init. Exposes version and label constants.
"""

__version__ = "1.0.0"
__author__  = "MTAH2027 Team"

# Label constants (matches submission format)
LABEL_HOME    = "H"
LABEL_AWAY    = "A"
LABEL_DEAD    = "D"
LABEL_UNKNOWN = "X"

# Integer codes used internally
CODE_HOME    = 0
CODE_AWAY    = 1
CODE_DEAD    = 2
CODE_UNKNOWN = 3

LABEL_TO_CODE = {LABEL_HOME: CODE_HOME, LABEL_AWAY: CODE_AWAY,
                 LABEL_DEAD: CODE_DEAD, LABEL_UNKNOWN: CODE_UNKNOWN}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}
