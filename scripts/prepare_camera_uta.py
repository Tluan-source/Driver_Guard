"""Compatibility entry point for the recovered week-5 module layout."""
import importlib
import sys

_module = importlib.import_module("scripts.data.prepare_camera_uta")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
