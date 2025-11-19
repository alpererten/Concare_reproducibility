"""
Instrumentation utilities for ConCare runs.
Provides signal recording hooks and perturbation evaluators that can be
plugged into train.py without bringing in external experiment trackers.
"""

from .manager import InstrumentationManager, InstrumentationConfig

__all__ = [
    "InstrumentationManager",
    "InstrumentationConfig",
]
