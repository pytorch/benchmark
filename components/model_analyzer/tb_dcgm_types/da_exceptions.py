class TorchBenchAnalyzerException(Exception):
    """
    A custom exception specific to the TorchBench Model Analyzer
    """
    pass


class TorchBenchAnalyzerExceptionGPUUnavailable(Exception):
    """
    A warning when the GPU is not visible to the process. 
    It is benign and can be ignored when there are multiple GPUs and only a subset of them are used.
    """

    def __init__(self, gpu_uuid='0000', *args: object) -> None:
        self.message = f'Warning: GPU with {gpu_uuid} uuid is not present!'
        super().__init__(self.message, *args)
