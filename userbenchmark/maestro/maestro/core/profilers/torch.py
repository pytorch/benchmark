import contextlib
from pathlib import Path
from typing import Optional

import torch

from core.profilers.base import profiler_ctx
from core.utils.logging import get_logger


logger = get_logger(__name__)

class TorchProfiler(profiler_ctx):
    """Doesn't work yet because get_results is not supported, to be fully functional the results dict should be populated in the _on_trace_ready callback"""
    def __init__(self, output_dir: Optional[Path] = None):
        raise NotImplementedError("TorchProfiler doesn't work yet.")

        super().__init__(output_dir)
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        )

    def get_results(self):
        raise NotImplementedError("TorchProfiler only support exporting trace to a file")
    
    def _save_trace(self, output_file: Path):
        self.profiler.export_chrome_trace(str(output_file.absolute()))

    def _enter(self):
        """Start torch profiling"""
        self.profiler.__enter__()
    
    def _exit(self, nosave: bool = False):
        """Stop torch profiling and optionally save results
        
        Args:
            nosave: If True, skip saving trace
        """
        self.profiler.__exit__(None, None, None)