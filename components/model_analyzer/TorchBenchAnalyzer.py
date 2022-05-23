
from pyexpat import model
from components.model_analyzer.dcgm.dcgm_monitor import DCGMMonitor
from components.model_analyzer.tb_dcgm_types.da_exceptions import TorchBenchAnalyzerException
from components.model_analyzer.tb_dcgm_types.gpu_device_factory import GPUDeviceFactory
from components.model_analyzer.dcgm import dcgm_fields
from components.model_analyzer.tb_dcgm_types.gpu_tensoractive import GPUTensorActive
from components.model_analyzer.tb_dcgm_types.gpu_utilization import GPUUtilization
from components.model_analyzer.tb_dcgm_types.gpu_power_usage import GPUPowerUsage
from components.model_analyzer.tb_dcgm_types.gpu_free_memory import GPUFreeMemory
from components.model_analyzer.tb_dcgm_types.gpu_used_memory import GPUUsedMemory
from components.model_analyzer.tb_dcgm_types.gpu_fp32active import GPUFP32Active
from components.model_analyzer.tb_dcgm_types.record_aggregator import RecordAggregator
import time
from components.model_analyzer.tb_dcgm_types.tb_logger import set_logger, LOGGER_NAME
from collections import defaultdict
from components.model_analyzer.tb_dcgm_types.config import *


import logging
logger = logging.getLogger(LOGGER_NAME)

class ModelAnalyzer:
    def __init__(self):
        self.gpu_factory = GPUDeviceFactory()
        self.gpus = self.gpu_factory.verify_requested_gpus(['all', ])
        # print(self.gpus)
        # the metrics to be collected
        self.gpu_metrics = [GPUUtilization, GPUPowerUsage,
                            GPUFreeMemory, GPUUsedMemory, GPUFP32Active, GPUTensorActive]
        # the final metric results. Its format is {GPU_UUID: {GPUUtilization: }}
        self.gpu_metric_value = {}
        self.gpu_monitor = None
        self.gpu_records = None
        self.config = AnalayzerConfig()
        set_logger()

    def start_monitor(self):
        try:
            self.gpu_monitor = DCGMMonitor(
                self.gpus, self.config.monitoring_interval, self.gpu_metrics)
            self.gpu_monitor.start_recording_metrics()
        except TorchBenchAnalyzerException:
            self._destory_monitor()
            raise

    def _destory_monitor(self):
        self.gpu_monitor.destroy()
        self.gpu_monitor = None
    
    def stop_monitor(self):
        """
        @return: collected gpu records @todo: add return type 
        """
        self.gpu_records = self.gpu_monitor.stop_recording_metrics()
        self._destory_monitor()
        return self.gpu_records
    
    def aggregate(self):
        gpu_record_aggregator = RecordAggregator()
        gpu_record_aggregator.insert_all(self.gpu_records)
        # @Yueming Hao todo: groupby GPU, then metric. Then calculate the flops
        records_groupby_gpu = {}
        records_groupby_gpu = gpu_record_aggregator.groupby(
            self.gpu_metrics, lambda record: record.device_uuid())
        
        for gpu in self.gpus:
            self.gpu_metric_value[gpu.device_uuid()] = {}
        for metric_type, metric in records_groupby_gpu.items():
            for gpu_uuid, metric_value in metric.items():
                self.gpu_metric_value[gpu_uuid][metric_type] = metric_value
        # return self.gpu_metric_value
    
    def set_monitoring_interval(self, attempted_interval):
        """
        The default monitoring internval is 10ms.
        """
        if attempted_interval < 0.001:
            logger.warning("The attempted interval is too short, would cause untruested profiling results.")
        self.config.monitoring_interval = attempted_interval

    def print_flops(self):
        print("==========Summary==========")
        for gpu_uuid in self.gpu_metric_value:
            gpu = self.gpu_factory.get_device_by_uuid(gpu_uuid)
            print(self.gpu_metric_value[gpu_uuid][GPUFP32Active].value())
            # TFLOPs/second = Device_SM_Count x Device_FMAs_Per_Cycle_Per_SM x 2 x Running_Frequency_KHz x DCGM_Activity / 1e+9
            print("GPU : TFLOPs/Second %.4f" % (gpu._sm_count * gpu._fma_count * 2 *
                gpu._frequency * self.gpu_metric_value[gpu_uuid][GPUFP32Active].value() / 1e+9))
        # @Yueming Hao: print all collected gpu records, for debug only
        # print(self.gpu_metrics)
        # for item in self.gpu_records:
        #     print(item)
        #     print(item.value())
    
    def calculate_flops(self, gpu_uuid=None):
        """
        The function to calculate TFLOPs/second for the desired GPU or the first available GPU.
        @return : a floating number representing TFLOPs/second.
        """
        if gpu_uuid:
            if gpu_uuid in self.gpu_metric_value:
                gpu = self.gpu_factory.get_device_by_uuid(gpu_uuid)
                return gpu._sm_count * gpu._fma_count * 2 * gpu._frequency * self.gpu_metric_value[gpu_uuid][GPUFP32Active].value() / 1e+9
            else:
                raise TorchBenchAnalyzerException("No available GPU with uuid ", gpu_uuid, " found!")
        else:
            if len(self.gpu_metric_value) > 1:
                logger.warning("There are multiple available GPUs and will only return the first one's flops.")
            gpu_uuid = next(iter(self.gpu_metric_value))
            gpu = self.gpu_factory.get_device_by_uuid(gpu_uuid)
            return gpu._sm_count * gpu._fma_count * 2 * gpu._frequency * self.gpu_metric_value[gpu_uuid][GPUFP32Active].value() / 1e+9


def run_app(n):
    import torch
    from torch.utils.benchmark import Timer

    x = torch.ones((n, n), device="cuda")
    y = torch.ones((n, n), device="cuda")
    torch.mm(x, y)

    # for n in [128, 1024, 4096, 8192]:
    # result = Timer(
    #     "torch.mm(x, y)",
    #     globals = {
    #         "x": torch.ones((n, n), device="cuda"),
    #         "y": torch.ones((n, n), device="cuda")}
    # ).blocked_autorange()
    # print(f"TFLOPs: {n**3 * 2 / result.median / 1e12:4>.1f}")
    # print(result, "\n")

    # import os
    # print(os.system(
    #     "/data/yhao/cuda-samples-master/Samples/0_Introduction/matrixMul.origin/matrixMul"))


