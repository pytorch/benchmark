
from .. import TorchBenchModelArgs

class TBExperimentBase:
    def __init__(self):
        pass
    
    @abstractmethod
    def get_model_config_iter(self):
        raise NotImplementedError

    @abstractmethod
    def get_runner(self):
        raise NotImplementedError

    @abstractmethod
    def get_reducer(self):
        raise NotImplementedError
