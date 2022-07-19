# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .monitor import Monitor
from ..tb_dcgm_types.gpu_free_memory import GPUFreeMemory
from ..tb_dcgm_types.gpu_tensoractive import GPUTensorActive
from ..tb_dcgm_types.gpu_used_memory import GPUUsedMemory
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


class DCGMMonitor(Monitor):
    """
    Use DCGM to monitor GPU metrics
    """

    # Mapping between the DCGM Fields and Model Analyzer Records
    # For more explainations, please refer to https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html
    model_analyzer_to_dcgm_field = {
        GPUUsedMemory: dcgm_fields.DCGM_FI_DEV_FB_USED,
        GPUFreeMemory: dcgm_fields.DCGM_FI_DEV_FB_FREE,
        GPUUtilization: dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
        GPUPowerUsage: dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
        GPUFP32Active: dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE,
        GPUTensorActive: dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
        GPUDRAMActive: dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
        GPUPCIERX: dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES,
        GPUPCIETX: dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES,
    }

    def __init__(self, gpus, frequency, metrics, dcgmPath=None):
        """
        Parameters
        ----------
        gpus : list of GPUDevice
            The gpus to be monitored
        frequency : int
            Sampling frequency for the metric
        metrics : list
            List of Record types to monitor
        dcgmPath : str (optional)
            DCGM installation path
        """

        super().__init__(frequency, metrics)
        structs._dcgmInit(dcgmPath)
        dcgm_agent.dcgmInit()

        self._gpus = gpus

        # Start DCGM in the embedded mode to use the shared library
        self.dcgm_handle = dcgm_handle = dcgm_agent.dcgmStartEmbedded(
            structs.DCGM_OPERATION_MODE_MANUAL)

        # Create DCGM monitor group
        self.group_id = dcgm_agent.dcgmGroupCreate(dcgm_handle,
                                                   structs.DCGM_GROUP_EMPTY,
                                                   "triton-monitor")
        # Add the GPUs to the group
        for gpu in self._gpus:
            dcgm_agent.dcgmGroupAddDevice(dcgm_handle, self.group_id,
                                          gpu.device_id())

        frequency = int(self._frequency * 1000)
        fields = []
        try:
            for metric in metrics:
                fields.append(self.model_analyzer_to_dcgm_field[metric])
        except KeyError:
            dcgm_agent.dcgmShutdown()
            raise TorchBenchAnalyzerException(
                f'{metric} is not supported by Model Analyzer DCGM Monitor')

        self.dcgm_field_group_id = dcgm_agent.dcgmFieldGroupCreate(
            dcgm_handle, fields, 'triton-monitor')

        self.group_watcher = dcgm_field_helpers.DcgmFieldGroupWatcher(
            dcgm_handle, self.group_id, self.dcgm_field_group_id.value,
            structs.DCGM_OPERATION_MODE_MANUAL, frequency, 3600, 0, 0)

    def _monitoring_iteration(self):
        self.group_watcher.GetMore()

    def _collect_records(self):
        records = []
        for gpu in self._gpus:
            device_id = gpu.device_id()
            metrics = self.group_watcher.values[device_id]

            # Find the first key in the metrics dictionary to find the
            # dictionary length
            if len(list(metrics)) > 0:
                for metric_type in self._metrics:
                    dcgm_field = self.model_analyzer_to_dcgm_field[metric_type]
                    for measurement in metrics[dcgm_field].values:

                        if measurement.value is not None:
                            # DCGM timestamp is in nanoseconds
                            records.append(
                                metric_type(value=float(measurement.value),
                                            device_uuid=gpu.device_uuid(),
                                            timestamp=measurement.ts))

        return records

    def destroy(self):
        """
        Destroy the DCGMMonitor. This function must be called
        in order to appropriately deallocate the resources.
        """

        dcgm_agent.dcgmShutdown()
        super().destroy()
