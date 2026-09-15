"""Home price estimation model (Phase 3)."""

import ctypes
import sys

if sys.platform == "win32":
    # pyarrow bundles its own (older) msvcp140.dll. If that copy loads first, LightGBM — built
    # against a newer MSVC STL — dereferences a null pointer and crashes with an access
    # violation the first time it constructs a Dataset. Loading the system copy here, before any
    # models.price submodule (and therefore before pandas/pyarrow) gets a chance to import,
    # makes every later load reuse this one. try/except so a missing system runtime never breaks
    # import.
    try:
        ctypes.WinDLL("msvcp140.dll")
    except OSError:
        pass
