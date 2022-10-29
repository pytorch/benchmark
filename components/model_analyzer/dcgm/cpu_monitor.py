import os
import time
from .monitor import Monitor
import psutil
from ..tb_dcgm_types.cpu_peak_memory import CPUPeakMemory

class CPUMonitor(Monitor):
    """
    A CPU monitor that uses psutil to monitor CPU usage
    """

    def __init__(self, frequency, metrics_needed=[]):
        super().__init__(frequency, metrics_needed)
        # It is a raw record list. [timestamp, cpu_memory_usage, cpu_available_memory]
        self._cpu_records = []
        # the current process is the process which launches and runs the deep learning models.
        self._monitored_pid = os.getpid()


    def _get_cpu_stats(self):
        """
        Append a raw record into self._cpu_metric_values.
        A raw record includes the timestamp in nanosecond, the CPU memory usage, CPU available memory in MB.
        """
        server_process = psutil.Process(self._monitored_pid)
        process_memory_info = server_process.memory_full_info()
        system_memory_info = psutil.virtual_memory()
        # Divide by 1024*1024 to convert from bytes to MB
        a_raw_record = (time.time_ns(), process_memory_info.uss // 1048576, system_memory_info.available // 1048576)
        return a_raw_record
        
    def _monitoring_iteration(self):
        if CPUPeakMemory in self._metrics:
            self._cpu_records.append(self._get_cpu_stats())

    def _collect_records(self):
        """
        Convert all raw records into corresponding Record type.
        """
        records = []
        for record in self._cpu_records:
            records.append(CPUPeakMemory(timestamp=record[0], value=record[1]))
        return records