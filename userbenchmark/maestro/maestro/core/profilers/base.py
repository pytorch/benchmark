from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

from core.utils.logging import get_logger
from core.utils.distributed import get_rank_and_world_size


logger = get_logger(__name__)

class profiler_ctx(ABC):
    """Profiler is a class that profiles a pattern
    
    Supports two usage patterns:
        1. Direct context manager: `with profiler:`
        2. Callable with flags: `with profiler(nosave=True):`
    """

    _saved_cnt = 0
    def __init__(self, output_dir: Optional[Union[Path, str]] = None):
        self.rank, self.world_size = get_rank_and_world_size()
        self._results = [] # History of results, every element is the results of a context iteration
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            logger.info(f"Profiler output directory: {self.output_dir}")
    
    def __call__(self, nosave: bool = False, name: str = None):
        """Return a context manager with custom flags
        
        Args:
            nosave: If True, skip saving trace on exit
            name: Name of this profiling session, will be used as a suffix for the trace file

        Usage:
            with profiler(nosave=True):
                # profiling code
        """
        return self._context(nosave=nosave, name=name)
    
    @contextmanager
    def _context(self, nosave: bool = False, name: str = None):
        """Internal context manager with flag support. Override in subclasses."""
        self._enter()
        try:
            yield self
        finally:
            self._exit()
            if not nosave:
                self.save_trace(file_suffix=name)
    
    def get_output_file(self, file_suffix: str = "") -> Path:
        if self.output_dir is None:
            return None
        stem = f"{self.__class__.__name__}_rank_{self.rank}_{self._saved_cnt}"
        if file_suffix:
            stem += f"_{file_suffix}"

        return self.output_dir / f"{stem}.trace"
    
    def _save_trace(self, output_file: Path):
        raise NotImplementedError(f"{self.__class__.__name__}._save_trace() not implemented")

    def save_trace(self, file_suffix: str = ""):
        output_file = self.get_output_file(file_suffix)
        if output_file is None:
            return
        try:
            self._save_trace(output_file)
        except Exception as e:
            logger.error(f"Error saving {self.__class__.__name__} trace to {output_file}: {e}")
        else:
            logger.debug(f"{self.__class__.__name__} trace saved to {output_file}")
        
        self._saved_cnt += 1
    
    def _save_results(self, results: list):
        mandatory_fields = ["stream_id", "duration_ns", "kind"]
        for result in results:
            for field in mandatory_fields:
                if field not in result:
                    raise ValueError(
                        f"Mandatory field '{field}' is not present in the result, probably a misimplementation of the profiler child-class"
                        )

        self._results.extend(results)
    
    def get_results(self) -> list[list[dict]]:
        return self._results
    
    def reset(self):
        self._results = []

    @abstractmethod
    def _enter(self):
        """Start profiling. Called by both __enter__ and _context."""
        pass

    @abstractmethod
    def _exit(self, nosave: bool = False):
        """Stop profiling and optionally save trace.
        
        Args:
            nosave: If True, skip saving trace
        """
        pass