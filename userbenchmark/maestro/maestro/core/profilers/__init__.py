"""Profilers are context managers that profile a pattern"""

from core.profilers.base import profiler_ctx
from core.profilers.torch import TorchProfiler
from core.profilers.cupti import CuptiProfiler
