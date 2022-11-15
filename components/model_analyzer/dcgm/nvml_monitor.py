import time
from .monitor import Monitor
from ..tb_dcgm_types.gpu_free_memory import GPUFreeMemory
from ..tb_dcgm_types.gpu_tensoractive import GPUTensorActive
from ..tb_dcgm_types.gpu_peak_memory import GPUPeakMemory
from ..tb_dcgm_types.gpu_utilization import GPUUtilization
from ..tb_dcgm_types.gpu_power_usage import GPUPowerUsage
from ..tb_dcgm_types.gpu_fp32active import GPUFP32Active
from ..tb_dcgm_types.gpu_dram_active import GPUDRAMActive
from ..tb_dcgm_types.gpu_pcie_rx import GPUPCIERX
from ..tb_dcgm_types.gpu_pcie_tx import GPUPCIETX
from ..tb_dcgm_types.da_exceptions import TorchBenchAnalyzerException


from . import dcgm_agent
from . import dcgm_fields
from . import dcgm_field_helpers
from . import dcgm_structs as structs




class NVMLMonitor(Monitor):
    """
    Use NVML to monitor GPU metrics
    """

    # Mapping between the NVML Fields and Model Analyzer Records
    # For more explainations, please refer to https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
    model_analyzer_to_nvml_field = {
        GPUPeakMemory: "used",
        GPUFreeMemory: "free",
        GPUUtilization: "utilization.gpu",
        GPUPowerUsage: "power.draw",
    }

    def __init__(self, gpus, frequency, metrics):
        """
        Parameters
        ----------
        gpus : list of GPUDevice
            The gpus to be monitored
        frequency : int
            Sampling frequency for the metric
        metrics : list
            List of Record types to monitor
        """

        super().__init__(frequency, metrics)
        import pynvml
        self._nvml = pynvml
        self._nvml.nvmlInit()
        self._metrics = metrics
        # raw records: {gpu: {field: [(timestamp, value), ...]}}
        self._records = {}
        self._gpus = gpus
        # gpu handles: {gpu: handle}
        self._gpu_handles = {}
        for gpu in self._gpus:
            self._gpu_handles[gpu] = self._nvml.nvmlDeviceGetHandleByUUID(gpu.device_uuid())
            self._records[gpu] = {}
            for metric in self._metrics:
                self._records[gpu][metric] = []

    def _monitoring_iteration(self):
        self._get_gpu_metrics()

    def _get_gpu_metrics(self):
        """
        Get the metrics of all the GPUs
        """
        for gpu in self._gpus:
            handle = self._nvml.nvmlDeviceGetHandleByUUID(gpu.device_uuid())
            for metric in self._metrics:
                nvml_field = self.model_analyzer_to_nvml_field[metric]
                atimestamp = time.time_ns()
                if metric == GPUPeakMemory:
                    info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    # @Yueming TODO: need to update with the nvml API version 2. Because the nvml API version 1 returns the used memory including the memory allocated by the GPU driver.
                    # used_mem = info.used
                    # reserved_mem = info.reserved
                    # self._records[gpu][metric].append((atimestamp, used_mem - reserved_mem))
                    self._records[gpu][metric].append((atimestamp, float(getattr(info, nvml_field)/1024/1024)))
                elif metric == GPUFreeMemory:
                    info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    self._records[gpu][metric].append((atimestamp, float(getattr(info, nvml_field)/1024/1024)))
                elif metric == GPUUtilization:
                    info = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    self._records[gpu][metric].append((atimestamp, getattr(info, nvml_field)))
                elif metric == GPUPowerUsage:
                    info = self._nvml.nvmlDeviceGetPowerUsage(handle)
                    self._records[gpu][metric].append((atimestamp, info))

    def _collect_records(self):
        records = []
        for gpu in self._gpus:
            for metric_type in self._metrics:
                for measurement in self._records[gpu][metric_type]:
                    records.append(metric_type(value=float(measurement[1]),
                                   timestamp=measurement[0], device_uuid=gpu.device_uuid()))
        return records
    
    def destroy(self):
        self._nvml.nvmlShutdown()
        super().destroy()
