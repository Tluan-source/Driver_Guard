"""Compatibility entry point for the recovered week-5 module layout."""
import importlib
import sys

_module = importlib.import_module("driverguard.learning.camera.model")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
