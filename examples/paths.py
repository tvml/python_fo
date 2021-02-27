"""Path hack to make tests work."""

import sys

if ".." not in sys.path:
    sys.path.insert(0, "..")
