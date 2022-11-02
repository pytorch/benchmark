import dataclasses
from typing import Optional, List, Dict, Any

@dataclasses.dataclass
class TorchBenchModelConfig:
    device: str
    test: str
    batch_size: Optional[int]
    jit: bool
    extra_args: List[str]

@dataclasses.dataclass
class TorchBenchModelMetrics:
    config: TorchBenchModelConfig
    status: str
    precision: str
    results: Dict[str, Any]
