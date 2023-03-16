
from typing import Optional, OrderedDict, Tuple

from .dcgm.cpu_monitor import CPUMonitor
from .dcgm.dcgm_monitor import DCGMMonitor
from .dcgm.nvml_monitor import NVMLMonitor
from .tb_dcgm_types.da_exceptions import TorchBenchAnalyzerException
from .tb_dcgm_types.gpu_device_factory import GPUDeviceFactory
from .dcgm import dcgm_fields
from .dcgm.dcgm_structs import DCGMError
from .tb_dcgm_types.gpu_tensoractive import GPUTensorActive
from .tb_dcgm_types.gpu_utilization import GPUUtilization
from .tb_dcgm_types.gpu_power_usage import GPUPowerUsage
from .tb_dcgm_types.gpu_free_memory import GPUFreeMemory
from .tb_dcgm_types.gpu_peak_memory import GPUPeakMemory
from .tb_dcgm_types.gpu_fp32active import GPUFP32Active
from .tb_dcgm_types.gpu_dram_active import GPUDRAMActive
from .tb_dcgm_types.gpu_pcie_rx import GPUPCIERX
from .tb_dcgm_types.gpu_pcie_tx import GPUPCIETX
from .tb_dcgm_types.cpu_peak_memory import CPUPeakMemory
from .tb_dcgm_types.record import RecordType
from .tb_dcgm_types.record_aggregator import RecordAggregator
from .tb_dcgm_types.tb_logger import set_logger, LOGGER_NAME
from .tb_dcgm_types.config import *

import logging
logger = logging.getLogger(LOGGER_NAME)
import json
from time import time_ns


class ModelAnalyzer:
    def __init__(self, export_metrics_file=None, metrics_needed=[], metrics_gpu_backend='nvml', cpu_monitored_pid=None):
        # For debug
        # set_logger(logging.DEBUG)
        set_logger()
        # delay the initialization to start_monitor
        self.gpu_factory = None
        self.gpus = None
        # the cpu metrics to be collected
        # self.gpu_metrics = [GPUUtilization, GPUPowerUsage,
        #                     GPUFreeMemory, GPUPeakMemory, GPUFP32Active, GPUTensorActive, GPUDRAMActive, GPUPCIERX, GPUPCIETX]
        self.gpu_metrics = []
        # the final metric results. Its format is {GPU_UUID: {GPUUtilization: }}
        # Example:
        # {'GPU-4177e846-1274-84e3-dcde':
        #   {<class '.tb_dcgm_types.gpu_fp32active.GPUFP32Active'>:
        #      <.tb_dcgm_types.gpu_fp32active.GPUFP32Active object at 0x7f14bbae2280>
        #   }
        #  }
        self.gpu_metric_value = {}
        # There are two kinds of GPU monitor: DCGMMonitor and NVMLMonitor
        self.gpu_monitor = None
        self.gpu_monitor_started = False
        self.gpu_records = None
        self.config = AnalayzerConfig()
        self.gpu_record_aggregator = RecordAggregator()
        self.export_csv_name = None
        self.set_export_csv_name(export_metrics_file)
        # the cpu metrics to be collected. available metrics are [CPUPeakMemory, ]
        self.cpu_metrics = []
        self.cpu_monitor = None
        self.cpu_monitor_started = False
        self.cpu_records = None
        self.cpu_record_aggregator = RecordAggregator()
        self.cpu_metric_value = {}
        self.cpu_monitored_pid = cpu_monitored_pid
        # GPU Monitor Backend
        self.gpu_monitor_backend = metrics_gpu_backend
        self.start_monitor_timestamp = None
        self.stop_monitor_timestamp = None
        self.metrics_backend_mapping = {}
        self.process_metrics(metrics_needed, metrics_gpu_backend)
    
    def process_metrics(self, metrics_needed, metrics_gpu_backend):
        if 'gpu_peak_mem' in metrics_needed:
            self.add_metric_gpu_peak_mem()
            self.metrics_backend_mapping['gpu_peak_mem'] = 'dcgm' if metrics_gpu_backend == 'dcgm' else 'nvml'
        if 'flops' in metrics_needed:
            if metrics_gpu_backend == 'dcgm':
                self.add_metric_gpu_flops()
                self.metrics_backend_mapping['flops'] = 'dcgm'
            else:
                self.metrics_backend_mapping['flops'] = 'fvcore'
        if 'cpu_peak_mem' in metrics_needed:
            self.add_metric_cpu_peak_mem()
        if metrics_gpu_backend == "default":
            self.set_gpu_monitor_backend_nvml()

    def add_metric_gpu_peak_mem(self):
        self.gpu_metrics.append(GPUPeakMemory)

    def add_metric_gpu_flops(self):
        self.gpu_metrics.append(GPUFP32Active)

    def add_metric_cpu_peak_mem(self):
        self.cpu_metrics.append(CPUPeakMemory)

    def set_gpu_monitor_backend_nvml(self):
        self.gpu_monitor_backend = 'nvml'

    def set_export_csv_name(self, export_csv_name=None):
        if not export_csv_name:
            return
        self.export_csv_name = export_csv_name
        # test for correct permission
        with open(export_csv_name, 'w') as fout:
            fout.write('')

    def update_export_name(self, insert_str=''):
        index = self.export_csv_name.find('.csv')
        if not index == -1:
            self.export_csv_name = self.export_csv_name[:index] + insert_str + self.export_csv_name[index:]

    def start_monitor(self):
        try:
            self.start_monitor_timestamp = time_ns()
            if self.gpu_metrics:
                self.gpu_factory = GPUDeviceFactory(self.gpu_monitor_backend)
                self.gpus = self.gpu_factory.verify_requested_gpus(['all', ])
                if not self.gpus:
                    raise TorchBenchAnalyzerException('No GPU found')
                if self.gpu_monitor_backend == 'dcgm':
                    self.gpu_monitor = DCGMMonitor(
                        self.gpus, self.config.monitoring_interval, self.gpu_metrics)
                elif self.gpu_monitor_backend == 'nvml':
                    self.gpu_monitor = NVMLMonitor(
                        self.gpus, self.config.monitoring_interval, self.gpu_metrics)
            if self.cpu_metrics:
                self.cpu_monitor = CPUMonitor(self.config.monitoring_interval, self.cpu_metrics, self.cpu_monitored_pid)
            if self.gpu_metrics:
                self.gpu_monitor.start_recording_metrics()
                self.gpu_monitor_started = True
            if self.cpu_metrics:
                self.cpu_monitor.start_recording_metrics()
                self.cpu_monitor_started = True
        except TorchBenchAnalyzerException:
            self._destory_monitor()
            raise

    def _destory_monitor(self):
        if self.gpu_monitor:
            self.gpu_monitor.destroy()
            self.gpu_monitor = None
            self.gpu_monitor_started = False
        if self.cpu_monitor:
            self.cpu_monitor.destroy()
            self.cpu_monitor = None
            self.cpu_monitor_started = False

    def stop_monitor(self):
        self.stop_monitor_timestamp = time_ns()
        if self.gpu_monitor:
            self.gpu_records = self.gpu_monitor.stop_recording_metrics()
        if self.cpu_monitor:
            self.cpu_records = self.cpu_monitor.stop_recording_metrics()
        # This must be called after stop_recording_metrics
        self._destory_monitor()

    def aggregate(self):
        """
        Aaggregate must be called after stop_monitor.
        """
        if self.gpu_records:
            new_gpu_records = [record for record in self.gpu_records if record.timestamp() <= self.stop_monitor_timestamp]
            if len(new_gpu_records) == 0:
                self.gpu_records = self.gpu_records[:1]
            else:
                self.gpu_records = new_gpu_records
            self.gpu_record_aggregator.insert_all(self.gpu_records)
            records_groupby_gpu = self.gpu_record_aggregator.groupby(
                self.gpu_metrics, lambda record: record.device_uuid())

            for gpu in self.gpus:
                self.gpu_metric_value[gpu.device_uuid()] = {}
            for metric_type, metric in records_groupby_gpu.items():
                for gpu_uuid, metric_value in metric.items():
                    self.gpu_metric_value[gpu_uuid][metric_type] = metric_value
        if self.cpu_records:
            new_cpu_records = [record for record in self.cpu_records if record.timestamp() <= self.stop_monitor_timestamp]
            if len(new_cpu_records) == 0:
                self.cpu_records = self.cpu_records[:1]
            else:
                self.cpu_records = new_cpu_records
            self.cpu_record_aggregator.insert_all(self.cpu_records)
            records_groupby_cpu = self.cpu_record_aggregator.groupby(
                self.cpu_metrics, lambda record: record.device_uuid())
            # detault cpu id is 0x1
            self.cpu_metric_value[0x1] = {}
            for metric_type, metric in records_groupby_cpu.items():
                for cpu_uuid, metric_value in metric.items():
                    self.cpu_metric_value[cpu_uuid][metric_type] = metric_value

    def set_monitoring_interval(self, attempted_interval):
        """
        The default monitoring internval is DEFAULT_MONITORING_INTERVAL * 1000 ms.
        """
        # if attempted_interval < 0.1:
        #     logger.warning("The attempted interval is too short, would cause untrusted profiling results.")
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
        logger.debug(json.dumps([_.to_dict() for _ in self.gpu_records], indent=4))

    def export_all_records_to_csv(self):
        records_groupby_gpu = self.gpu_record_aggregator.groupby_wo_aggregate(
            self.gpu_metrics, lambda record: record.device_uuid())
        # {GPUUUID: {record_type: {timestamp: a_record, } }}
        csv_records = {}
        for gpu in self.gpus:
            csv_records[gpu.device_uuid()] = OrderedDict()
        for record_type in records_groupby_gpu:
            csv_records[gpu.device_uuid()][record_type] = OrderedDict()
            for gpu_uuid in records_groupby_gpu[record_type]:
                cluster_records = records_groupby_gpu[record_type][gpu_uuid][record_type]
                cluster_records.sort(key=lambda x: x.timestamp())
                for record in cluster_records:
                    csv_records[gpu_uuid][record_type][record.timestamp()] = record.value()
        with open(self.export_csv_name, 'w') as fout:
            for gpu_uuid in csv_records:
                # timestamp record in DCGM is microsecond
                timestamps = set()
                fout.write("timestamp(ms), ")
                for record_type in csv_records[gpu_uuid]:
                    timestamps |= set(csv_records[gpu_uuid][record_type])
                    if record_type.tag == "gpu_fp32active":
                        tmp_line = "%s, " % (record_type.tag + '(%)')
                    elif record_type.tag.startswith('gpu_pice'):
                        tmp_line = "%s, " % (record_type.tag + '(bytes)')
                    elif record_type.tag == 'gpu_peak_memory':
                        tmp_line = "%s, " % (record_type.tag + '(MB)')
                    else:
                        tmp_line = "%s, " % record_type.tag
                    fout.write(tmp_line)
                fout.write("duration(ms), ")
                if GPUPCIERX in self.gpu_metrics:
                    fout.write("HtoD_throughput(GB/s), ")
                if GPUPCIETX in self.gpu_metrics:
                    fout.write("DtoH_throughput(GB/s), ")
                timestamps = list(timestamps)
                timestamps.sort()
                timestamp_start = timestamps[0]
                fout.write('\n')
                last_timestamp = timestamp_start
                for a_timestamp in timestamps:
                    duration = (a_timestamp - last_timestamp) / 1e3
                    last_timestamp = a_timestamp
                    line = "%.2f, " % ((a_timestamp - timestamp_start) / 1000)
                    for record_type in csv_records[gpu_uuid]:
                        value = csv_records[gpu_uuid][record_type].get(a_timestamp, -1)
                        line += "%.2f, " % value
                    line += "%.2f, " % duration
                    if duration != 0 :
                        if GPUPCIERX in self.gpu_metrics:
                            pcierx_record = csv_records[gpu_uuid][GPUPCIERX].get(a_timestamp, -1)
                            if pcierx_record != -1:
                                line += "%.2f, " % (pcierx_record / duration * 1000 / 1024 / 1024 / 1024)
                        if GPUPCIETX in self.gpu_metrics:
                            pcietx_record = csv_records[gpu_uuid][GPUPCIETX].get(a_timestamp, -1)
                            line += "%.2f, " % (pcietx_record / duration * 1000 / 1024 / 1024 / 1024)
                    fout.write(line + "\n")

    def calculate_flops(self, gpu_uuid=None) -> float:
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
            # Will only return the first one's peak memory bandwidth. So please use CUDA_VISIBLE_DEVICES to specify the GPU.
            gpu_uuid = next(iter(self.gpu_metric_value))
            gpu = self.gpu_factory.get_device_by_uuid(gpu_uuid)
            device_id = self.gpu_factory.get_device_by_uuid(gpu_uuid).device_id()
            return device_id, gpu._sm_count * gpu._fma_count * 2 * gpu._frequency * self.gpu_metric_value[gpu_uuid][GPUFP32Active].value() / 1e+9

    def calculate_gpu_peak_mem(self, gpu_uuid=None) -> Tuple[Optional[str], float]:
        """
        The function to calculate GPU peak memory usage for the first available GPU.
        @return : a floating number representing GB.
        """
        if gpu_uuid:
            if gpu_uuid in self.gpu_metric_value:
                return self.gpu_metric_value[gpu_uuid][GPUPeakMemory].value() / 1024
            else:
                raise TorchBenchAnalyzerException("No available GPU with uuid ", gpu_uuid, " found!")
        if len(self.gpu_metric_value) == 0:
            raise TorchBenchAnalyzerException("No metrics collected!")
        # Will only return the first one's peak memory bandwidth. So please use CUDA_VISIBLE_DEVICES to specify the GPU.
        gpu_uuid = next(iter(self.gpu_metric_value))
        device_id = self.gpu_factory.get_device_by_uuid(gpu_uuid).device_id()
        return device_id, self.gpu_metric_value[gpu_uuid][GPUPeakMemory].value() / 1024

    def calculate_cpu_peak_mem(self, cpu_uuid=None) -> float:
        """
        The function to calculate CPU peak memory usage.
        @return : a floating number representing GB.
        """
        if len(self.cpu_metric_value) > 1:
            logger.debug("There are multiple available CPUs and will only return the first one's peak memory bandwidth.")
        cpu_uuid = next(iter(self.cpu_metric_value))
        return self.cpu_metric_value[cpu_uuid][CPUPeakMemory].value() / 1024
    




def check_dcgm():
    try:
        temp_model_analyzer = ModelAnalyzer()
        temp_model_analyzer.add_metric_gpu_flops()
        temp_model_analyzer.start_monitor()
        temp_model_analyzer.stop_monitor()
    except DCGMError as e:
        logger.error("ERROR: DCGM init failed. ", e)
        exit(-1)
    return True


def check_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.error("ERROR: NVML init failed. Please check the installation of pynvml.", e)
        exit(-1)
    return True
