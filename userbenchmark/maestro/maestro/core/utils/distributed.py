from abc import ABC, abstractmethod
import time
import os
from typing import Optional, Any

import torch
import torch.distributed as torch_dist

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

        
class _TorchDistUtils(_DistUtils):
    def __init__(self, backend=None):
        self.backend = backend
        super().__init__()
    
    @classmethod
    def get_world_group(cls):
        return torch_dist.group.WORLD
    
    def get_device(self) -> torch.device:
        visible_devices = torch.cuda.device_count()
        if visible_devices == 0:
            raise RuntimeError("No CUDA devices are visible")
        # SLURM may give each task one visible GPU, whose CUDA ordinal is zero.
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        device_index = 0 if visible_devices == 1 else local_rank
        if not 0 <= device_index < visible_devices:
            raise ValueError(f"Local rank {local_rank} is outside {visible_devices} visible CUDA devices")
        return torch.device(f"cuda:{device_index}")

    def _init_dist(self):
        """Init torch distributed module"""

        rank, world_size = get_rank_and_world_size()

        # Init device - pytorch recommends to set it with CUDA_VISIBLE_DEVICES
        # but for our usecase we will just assign every rank to a different device
        # of course ranks/node should be equal to devices/node
        device = self.get_device()
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        logger.debug(f"Rank {rank} is on device {device} ({torch.cuda.device_count()} devices available)")

        torch_dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
            device_id=device,       # Passing device ID triggers nccl comm initialization
        )

        if rank == 0:
            logger.debug(f"Initialized torch distributed with {world_size} ranks")

    
    def barrier(self, async_op: bool = False) -> Optional[AsyncWorkHandle]:
        work = torch_dist.barrier(async_op=async_op)
        if async_op:
            return AsyncWorkHandle(work, type=AsyncWorkHandleType.TORCH)
        return None

    def get_pg_opts(self, params: dict = None) -> dict:
        return None

    def create_axis(self, axis: list[list[int]], params: dict = None) -> "torch_dist.ProcessGroup":
        my_group = None
        pg_opts = self.get_pg_opts(params)

        for ranks in axis:
            options = {"ranks": ranks, "backend": self.backend, "pg_options": pg_opts}
            if self.backend == "nccl":
                # Bind NCCL eagerly to this process's selected CUDA device.
                options["device_id"] = self.get_device()
            pg = torch_dist.new_group(**options)
            if pg != torch_dist.GroupMember.NON_GROUP_MEMBER:
                my_group = pg

        return my_group
    
    def get_rank(self):
        return torch_dist.get_rank()

    def get_world_size(self):
        return torch_dist.get_world_size()
    
    def all_gather_object(self, obj: Any) -> list[Any]:
        output = [None]*self.get_world_size()
        torch_dist.all_gather_object(output, obj)
        return output
    
    def destroy(self):
        try:
            torch_dist.destroy_process_group()
        except:
            # Probably already destroyed
            pass

class _TorchNCCLDistUtils(_TorchDistUtils):
    def __init__(self):
        super().__init__(backend="nccl")

    def get_pg_opts(self, params: Optional[dict] = None) -> dict:
        if not params:
            return None
        opts = torch_dist.ProcessGroupNCCL.Options()
        if "traffic_class" in params:
            opts.config.traffic_class = params["traffic_class"]
        return opts



try:
    nccl_torch_dist_utils = _TorchNCCLDistUtils()
except Exception as e:
    logger.error(f"Error initializing NCCL distributed utilities: {e}")
    raise
