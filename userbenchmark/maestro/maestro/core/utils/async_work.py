from abc import ABC
from typing import Any, Optional
from enum import Enum


class AsyncWorkHandleType(Enum):
    TORCH = "torch"

class AsyncWorkHandle(ABC):
    """Generic async work handle wrapper."""

    def __init__(self, work: Any, type: Optional[AsyncWorkHandleType] = None):
        """
        name: Name of the work handle, optional, to identify the work handle type as it is a generic wrapper 
        """
        self.work = work
        self.type = type
    
    def block_current_stream(self):
        """Block the current GPU stream until the work is completed"""
        if self.type == AsyncWorkHandleType.TORCH:
            self.work.block_current_stream()
        else:
            raise ValueError(f"Block current stream is not supported for work handle of type {self.type}")