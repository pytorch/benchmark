from abc import ABC, abstractmethod
from typing import Iterator, ContextManager
from contextlib import contextmanager
import contextvars
import torch

from core.utils.distributed import nccl_torch_dist_utils
from core.utils.logging import get_logger


logger = get_logger(__name__)

# Always contain the current axis used
current_axis = contextvars.ContextVar("current_axis", default=None)


class Axis(ABC):

    dist_utils = None

    def __init__(self, groups: list[list[int]], name: str = ""):
        """
        Axis is a set of teams created using the same pattern. Each axis can execute blocks independently from the other axes (i.e it is bound to a GPU single stream)
        For now, each rank must be in one group of the axis
        """
        if self.__class__.dist_utils is None:
            raise ValueError(f"class attribute dist_utils is not set for {self.__class__.__name__}")

        self.groups = groups
        self.name = name

        self.my_group = self.dist_utils.create_axis(groups)
        r = self.get_rank()
        for group in self.groups:
            if r in group:
                self.my_group_ranks = group
                break
        
        if not self.my_group_ranks:
            raise NotImplementedError(f"Rank {r} is not in any group of axis {self.name}")
        
        self.root_rank = min(self.my_group_ranks)
    
    def team_size(self):
        return len(self.groups[0])
    
    @classmethod
    def get_rank(cls):
        return cls.dist_utils.get_rank()

    @classmethod
    def get_world_size(cls):
        return cls.dist_utils.get_world_size()
    
    @contextmanager
    def use_axis(self, block_current_stream=False) -> Iterator[None]:
        """Context manager to use the axis"""
        token = current_axis.set(self)
        try:
            with self._axis_ctx():
                yield
        finally:
            current_axis.reset(token)
    
    @abstractmethod
    def _axis_ctx(self) -> ContextManager[None]:
        """Wrapper around a context manager to use the axis"""
        pass

    @staticmethod
    @abstractmethod
    def create_event():
        """Create a new event of the type expected by record_event and wait_event"""
        pass

    @staticmethod
    @abstractmethod
    def record_event(event):
        """Record the event on the axis
        Not bound to any axis - just record the event on the current stream
        """
        pass
    
    @abstractmethod
    def wait_event(self, event):
        """Wraps cudaStreamWaitEvent - Block this axis until the event is recorded by record_event"""
        pass
    
    @abstractmethod
    def synchronize(self):
        """Wrapper around cudaStreamSynchronize - Block CPU until all the blocks on this axis have finished"""
        pass
    
    @classmethod
    @abstractmethod
    def synchronize_all(cls):
        """Wrapper around cudaDeviceSynchronize - Block CPU until all GPU streams have finished"""
        pass
    
    @classmethod
    def destroy(cls):
        """Destroy the axis"""
        pass


class TorchAxis(Axis):
    dist_utils = nccl_torch_dist_utils

    def __init__(self, groups: list[list[int]], name: str = ""):

        super().__init__(groups, name=name)
        self.stream = torch.cuda.Stream()
    
    def get_process_group(self) -> torch.distributed.ProcessGroup:
        return self.my_group
    
    @staticmethod
    def record_event(event: torch.cuda.Event):
        event.record()

    @staticmethod
    def create_event() -> torch.cuda.Event:
        return torch.cuda.Event()
    
    def _axis_ctx(self):
        return torch.cuda.stream(self.stream)

    def wait_event(self, event: torch.cuda.Event):
        self.stream.wait_event(event)
    
    def synchronize(self):
        self.stream.synchronize()
    
    @classmethod
    def synchronize_all(cls):
        torch.cuda.synchronize()
    
    @classmethod
    def destroy(cls):
        """Destroy the axis"""
        cls.dist_utils.destroy()