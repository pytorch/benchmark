from abc import ABC, abstractmethod
import time
import os
from typing import Optional, Any

from .async_work import AsyncWorkHandle, AsyncWorkHandleType

# Has to be above import of get_logger to avoid circular import
def get_rank_and_world_size():
    """Parse rank and world size from environment variables"""
    # torchrun variables take precedence when launched inside a SLURM allocation.
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        return 0, 1

    return rank, world_size


from .logging import get_logger
logger = get_logger(__name__)


def create_axis_by_stride(world_size, stride, size):
    unmatched_ranks = list(range(world_size))
    axis = []
    while unmatched_ranks:
        team = []
        axis.append(team)
        r = min(unmatched_ranks)
        for _ in range(size):
            team.append(r)
            try:
                unmatched_ranks.remove(r)
            except ValueError:
                raise ValueError(f"Axis stride {stride} and size {size} are incompatible with world size {world_size}")
            r += stride

    for t in axis:
        if len(t) != size:
            raise ValueError(f"Axis stride {stride} and size {size} are incompatible with world size {world_size}")

    return axis

def dist_print(*args, **kwargs):
    """Print a message with the rank prefix"""
    rank, world_size = get_rank_and_world_size()
    print(f"[Rank {rank}]", *args, **kwargs)

def root_print(*args, **kwargs):
    """Print a message only if rank is 0"""
    rank, world_size = get_rank_and_world_size()
    if rank == 0:
        print(*args, **kwargs)

def dist_breakpoint(other_sleep=True):
    """Breakpoint only if rank is 0"""
    rank, world_size = get_rank_and_world_size()
    if rank == 0:
        breakpoint()
    elif other_sleep:
        time.sleep(1000)


class _DistUtils(ABC):
    def __init__(self):
        self._is_init = False
        self.init_dist()

    def init_dist(self):
        if self._is_init:
            return self.get_rank(), self.get_world_size()
        self._init_dist()
        self._is_init = True
        return self.get_rank(), self.get_world_size()

    @classmethod
    @abstractmethod
    def get_world_group(cls):
        pass

    @abstractmethod
    def _init_dist(self):
        """Inner method that actually initializes the distributed module"""
        pass

    @abstractmethod
    def get_rank(self):
        pass

    @abstractmethod
    def get_world_size(self):
        pass

    @abstractmethod
    def create_axis(self, axis: list[list[int]], params: dict = None):
        """Create an axis from a list of lists of ranks and return the group this rank is in"""
        pass

    # Logging utils
    def print(self, *args, **kwargs):
        if self.get_rank() == 0:
            print(*args, **kwargs)

    def info(self, *args, **kwargs):
        if self.get_rank() == 0:
            logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.get_rank() == 0:
            logger.debug(*args, **kwargs)

    def warn(self, *args, **kwargs):
        if self.get_rank() == 0:
            logger.warn(*args, **kwargs)


    @abstractmethod
    def all_gather_object(self, object: Any) -> list[Any]:
        """All gather an object across all the ranks"""
        pass

    @abstractmethod
    def barrier(self, async_op: bool = False) -> Optional["AsyncWorkHandle"]:
        """Barrier all the ranks"""
        pass
