"""Compatibility bridge to the unified factor directory."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("factors.reference.market_breadth")
