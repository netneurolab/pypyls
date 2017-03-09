import sys

if sys.version_info < (3,2):
    raise Exception("Python version needs to be >= 3.5")

import pyls.compute
