import dataclasses
import gc
import threading

from pathlib import Path
from typing import Dict, List, Optional

from components._impl.tasks import base as base_task
from components._impl.workers import subprocess_worker

from torchbenchmark import ModelDetails, Worker


@dataclasses.dataclass
class TBUserbenchmarkConfig:
    name: str
    args: List[str]
    output_dir: Optional[Path] = None

    @property
    def output_dir_name(self) -> str:
        return self.name + " " + " ".join(self.args)

class TBUserTask(base_task.TaskBase):

    # The worker may (and often does) consume significant system resources.
    # In order to ensure that runs do not interfere with each other, we only
    # allow a single UserTask to exist at a time.
    _lock = threading.Lock()

    def __init__(
        self,
        config: TBUserbenchmarkConfig,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        # gc.collect()  # Make sure previous task has a chance to release the lock
        assert self._lock.acquire(blocking=False), "Failed to acquire lock."
        self._worker = Worker(timeout=timeout, extra_env=extra_env, save_output_dir=config.output_dir)
        self._details = config
        self._maybe_import_userbenchmark(config.name)

    def __del__(self) -> None:
        self._lock.release()

    @property
    def worker(self) -> subprocess_worker.SubprocessWorker:
        return self._worker

    def __str__(self) -> str:
        return f"TBUserTask(Name: {self._details.name}, Details: {self._details})"

    # =========================================================================
    # == Run the userbenchmark in subprocess   ================================
    # =========================================================================
    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def _maybe_import_userbenchmark(ub_name: str) -> None:
        import importlib
        import os
        import traceback
        try:
            module = importlib.import_module(f'.{ub_name}.run', package="userbenchmark")
            run_func = getattr(module, 'run', None)
        except ModuleNotFoundError:
            traceback.print_exc()
            exit(-1)

        # Populate global namespace so subsequent calls to worker.run can access `Model`
        globals()["_run_func"] = run_func
        return

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def run(args: List[str]) -> None:
        import gc
        gc.collect()
        run_func = globals()["_run_func"]
        run_func(args)
