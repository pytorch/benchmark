"""
Measure the performance stableness of a model
"""

import dataclasses
from typing import List
from .base import ExperimentBase

@dataclasses.dataclass
class Metric:
    p0: float
    p50: float
    p100: float
    stdev: float

class Experiment(ExperimentBase):
    def __init__(self) -> None:
        super().__init__()
        self.metrics: List[Metric] = []

    def run(self):
        pass