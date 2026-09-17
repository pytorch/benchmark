from abc import ABC, abstractmethod, ABCMeta
from typing import Optional
import torch
from core.utils.logging import get_logger, get_root_rank_logger
from core.axis import Axis
from core.utils.format import format_size

from core.axis import TorchAxis

logger = get_logger(__name__)
rlogger = get_root_rank_logger()

class BlockRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, block: type):
        registry_name = getattr(block, "registry_name", None)
        if not registry_name:
            logger.debug(f"Not registering Block class {block.__name__} because it doesn't have the registry_name attribute")
            return
        if registry_name in cls._registry:
            raise ValueError(f"Block {registry_name} already registered")

        cls._registry[registry_name] = block
    
    @classmethod
    def get_block_cls(cls, name: str) -> type:
        """Return the block class with this name"""
        if name not in cls._registry:
            raise NameError(f"Could not find block with name {name}, registered: {list(cls._registry.keys())}")
        block_cls = cls._registry[name]
        return block_cls
    
    @classmethod
    def get_all_blocks(cls) -> list[type]:
        return list(cls._registry.values())


class _BlockRegistrer(ABCMeta):
    registry = {}
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        # Don't register abstract classes
        if getattr(cls, "__abstractmethods__", False):
            return
        
        logger.debug(f"[_BlockRegistry] Registering block {cls}")
        BlockRegistry.register(cls)


class _Block(ABC, metaclass=_BlockRegistrer):
    """Each block should have a 'registry_name' class variable"""
    _instances_cnt = 0
    registry_name: str = ""

    api_params_schema: dict[str, dict] = {} 
    """
    api_params_schema: Dict param_name: constraints where constraints is a dict that can implement:
        - 'default': A default value, any param without default is considered required
    Example:
    api_params_schema = {
        'axis': {},
        'tp': {'default': 1} 
    }
    """
    def __init__(self, axis: "Axis", name: str = ""):

        # Known limitation of some blocks that are blocking the other streams, warn the user
        if getattr(self, "IS_BLOCKING_CPU_BLOCK", False):
            rlogger.warning(f"{self.__class__.__name__} is a blocking CPU block, it will block the other streams, put it at the end of the pattern if possible, or be aware that the blocks that come after won't overlap")

        self.__class__._instances_cnt += 1
        self.axis = axis
        self.name = name

        if not self.name:
            self.name = f"{self.__class__.__name__} {self.__class__._instances_cnt}"
        

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._instances_cnt = 0

    @abstractmethod
    def run(self):
        """Start a GPU kernel"""
        pass
    
    @classmethod
    @abstractmethod
    def get_tensor_size(cls, api_params: dict) -> int:
        """Return the size of the tensor required to run this block, -1 means no tensor is needed
        Params: 
            api_params: The parameters required to run this block, following api_params_schema
        """
        pass

    @classmethod
    def _enrich_api_params_schema(cls, params: dict):
        """Validate and enrich the parameters required to create the block"""
        missing = []
        for param, constraints in cls.api_params_schema.items():
            if param not in params:
                if 'default' in constraints:
                    params[param] = constraints['default']
                else:
                    missing.append(param)
        if missing:
            raise ValueError(f"Required param {missing} are required to create the block {cls.__name__}, got: {params}")
        
        return params
    
    @classmethod
    @abstractmethod
    def from_api_params(cls, axis: "Axis", name: str, api_params: dict, cached_tensor: Optional[torch.Tensor] = None):
        """Create a block from the api parameters
        Params:
        api_params: The parameters required to create the block, following cls.api_params_schema
        cached_tensor: A tensor that can be used to create the block, if not provided, a new tensor will be created
        """
        pass

    def get_block_size_str(self) -> str:
        """Return the size of the block in a human readable string"""
        return "N/A"


class CommBlock(_Block):
    """Communication block"""

    api_params_schema = {"vector_size": {}}

    def __init__(self, axis: "Axis", src_buf, dst_buf, name: str = ""):
        super().__init__(axis, name)
        self.src_buf = src_buf
        self.dst_buf = dst_buf
    
    @classmethod
    def get_src_buf_size(cls, full_vector_size: int, team_size: int):
        return full_vector_size

    @classmethod
    def get_dst_buf_size(cls, full_vector_size: int, team_size: int):
        return full_vector_size
    
    @staticmethod
    def get_bus_bw_factor(team_size: int):
        """See NCCL tests PERFORMANCE.md for more details"""
        return 1

    def get_block_size_str(self) -> str:
        return format_size(self.get_full_vector_size())

    def get_full_vector_size(self) -> int:
        return self._get_full_vector_count() * self._get_element_size()
    
    def _get_element_size(self) -> int:
        """Change accordingly only if src_buf and dst_buf have different element sizes"""
        return self.src_buf.dtype.itemsize

    def _get_full_vector_count(self) -> int:
        """Full vector size is the size that is considered to be the 'size' of the communication, by default it is the max of the src and dst buffer sizes
        For example, in NCCL-tests the size parameter of the all_gather test is the size of the source vector, but actually 
        the data sent is nranks times this size, this nranks*size is considered the full vector size.
        """
        return max(self.src_buf.shape[0], self.dst_buf.shape[0])
    
    def get_alg_bw(self, exec_time_sec: float) -> float:
        """Returns alg bw in GB/s"""
        if exec_time_sec == 0:
            return 0

        total_msg_size_gb = self.get_full_vector_size() / 1E9
        alg_bw = total_msg_size_gb / exec_time_sec
        return alg_bw
    
    def get_bus_bw(self, exec_time_sec: float):
        """Returns bus bw in GB/s"""
        alg_bw = self.get_alg_bw(exec_time_sec)
        bus_bw = alg_bw * self.get_bus_bw_factor(self.axis.team_size())
        return bus_bw
    
    @classmethod
    def get_tensor_size(cls, api_params: dict) -> int:
        return api_params["vector_size"]
    
    @classmethod
    def from_api_params(cls, axis: "Axis", name: str, api_params: dict, cached_tensor: Optional[torch.Tensor] = None):
        src_buf_size = cls.get_src_buf_size(api_params["vector_size"], axis.team_size())
        dst_buf_size = cls.get_dst_buf_size(api_params["vector_size"], axis.team_size())

        if cached_tensor is None:
            cached_tensor = torch.zeros(cls.get_tensor_size(api_params), device="cuda")

        src_buf = cached_tensor[:src_buf_size]
        dst_buf = cached_tensor[:dst_buf_size]
        return cls(axis=axis, src_buf=src_buf, dst_buf=dst_buf, name=name)


class ComputeBlock(_Block):
    """Compute block"""
    pass


class GEMMBlock(ComputeBlock):
    """GEMM block"""

    api_params_schema = {"mat_a_shape": {}, "mat_b_shape": {}}

    def __init__(self, 
        axis: "Axis", 
        mat_a,
        mat_b,
        name: str = "",
        ):
        """
        Args:
            axis: Axis object for process group management
            name: Optional name for the block
            mat_a: Input matrix A (M x K)
            mat_b: Input matrix B (K x N)
        """
        super().__init__(axis, name)
        self.mat_a = mat_a  
        self.mat_b = mat_b  

    def get_block_size_str(self) -> str:
        return f"({self.mat_a.shape[0]}x{self.mat_a.shape[1]})x({self.mat_b.shape[0]}x{self.mat_b.shape[1]})"
    
    @classmethod
    def get_tensor_size(cls, api_params: dict) -> int:

        size_a = api_params["mat_a_shape"][0] * api_params["mat_a_shape"][1]
        size_b = api_params["mat_b_shape"][0] * api_params["mat_b_shape"][1]
        return size_a + size_b 
    
    @classmethod
    def from_api_params(cls, axis: "Axis", name: str, api_params: dict, cached_tensor: Optional[torch.Tensor] = None):
        if cached_tensor is None:
            cached_tensor = torch.zeros(cls.get_tensor_size(api_params), device="cuda")

        size_a = api_params["mat_a_shape"][0] * api_params["mat_a_shape"][1]
        size_b = api_params["mat_b_shape"][0] * api_params["mat_b_shape"][1]
        mat_a = cached_tensor[:size_a].reshape(api_params["mat_a_shape"][0], api_params["mat_a_shape"][1])
        mat_b = cached_tensor[size_a:size_a+size_b].reshape(api_params["mat_b_shape"][0], api_params["mat_b_shape"][1])

        return cls(axis=axis, mat_a=mat_a, mat_b=mat_b, name=name)



class CopyBlock(_Block):
    """Copy block"""
    pass


class TorchNNModuleBlock(_Block):
    """Wrapper for torch.nn.Module"""
    def __init__(self, axis: "Axis", input_tensor: torch.Tensor, name: str = ""):
        if not isinstance(axis, TorchAxis):
            raise ValueError(f"{self.__class__.__name__} axis should be a TorchAxis")
        
        super().__init__(axis, name)
        self.input_tensor = input_tensor
        self.module: torch.nn.Module = None
    
    @abstractmethod
    def _run(self):
        """Run the module's function"""
        pass

    def run(self):
        if self.module is None:
            raise ValueError(f"self.module has not been initialized for {self.__class__.__name__}")
        
        return self._run()


class MegatronModuleBlock(TorchNNModuleBlock):
    pass

# Load all the custom blocks
import blocks