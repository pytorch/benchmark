# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import logging
from numba.cuda.cudadrv import enums
# @Yueming Hao: TODO: Replace this with nvml API
from numba import cuda
from .da_exceptions import TorchBenchAnalyzerException, TorchBenchAnalyzerException_GPU_Inavailable



class Device:
    """
    Generic device class representing devices being monitored
    """

    def __init__(self):
        pass


class GPUDevice(Device):
    """
    Representing a GPU device
    """

    def __init__(self, device_name, device_id, pci_bus_id, device_uuid):
        """
        Parameters
        ----------
            device_name: str
                Human readable name of the device
            device_id : int
                Device id according to the `nvidia-smi` output
            pci_bus_id : str
                PCI bus id
            device_uuid : str
                Device UUID
        """

        assert type(device_name) is str
        assert type(device_id) is int
        assert type(pci_bus_id) is str
        assert type(device_uuid) is str

        self._device_name = device_name
        self._device_id = device_id
        self._pci_bus_id = pci_bus_id
        self._device_uuid = device_uuid
        self._device = None
        self._sm_count = 0
        for gpu in cuda.gpus:
            if gpu._device.uuid == device_uuid:
                self._device = gpu
        if self._device is None:
            raise TorchBenchAnalyzerException_GPU_Inavailable(device_uuid)

        self._sm_count = self._device.MULTIPROCESSOR_COUNT
        fma_count = ConvertSMVer2Cores(self._device.COMPUTE_CAPABILITY_MAJOR, self._device.COMPUTE_CAPABILITY_MINOR)
        if fma_count == 0:
            raise TorchBenchAnalyzerException('Unsupported GPU arch with CC%d.%d. Please check ConvertSMVer2Cores function.'
             %(self._device.COMPUTE_CAPABILITY_MAJOR, self._device.COMPUTE_CAPABILITY_MINOR))
        self._fma_count = fma_count
        self._frequency = self._device.CLOCK_RATE

    def device_name(self):
        """
        Returns
        -------
        str
            device name
        """

        return self._device_name

    def device_id(self):
        """
        Returns
        -------
        int
            device id of this GPU
        """

        return self._device_id

    def pci_bus_id(self):
        """
        Returns
        -------
        bytes
            PCI bus id of this GPU
        """

        return self._pci_bus_id

    def device_uuid(self):
        """
        Returns
        -------
        str
            UUID of this GPU
        """

        return self._device_uuid

    def sm_count(self):
        """
        Returns
        -------
        int
            number of SMs on this GPU
        """
        return self._sm_count

def ConvertSMVer2Cores(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    # Refer to https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
    return {(3, 0): 192,  # Kepler
            (3, 2): 192,
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,  # Maxwell
            (5, 2): 128,
            (5, 3): 128,
            (6, 0): 64,   # Pascal
            (6, 1): 128,
            (6, 2): 128,
            (7, 0): 64,   # Volta
            (7, 2): 64,
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere
            (8, 6): 128,
            (8, 7): 128
            }.get((major, minor), 0)

