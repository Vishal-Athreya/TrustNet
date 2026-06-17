# __init__.py
# Exposes only the public API of this module.
# Backend imports run_behavior_analysis — nothing else is needed externally.

from .main import run_behavior_analysis

__all__ = ["run_behavior_analysis"]