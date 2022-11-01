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

# Logging
LOGGER_NAME = "model_analyzer_logger"
from .gpu_device import GPUDevice
from  ..dcgm import dcgm_agent as dcgm_agent
from ..dcgm import dcgm_structs as structs
from .da_exceptions import TorchBenchAnalyzerException

import numba.cuda
import logging

logger = logging.getLogger(LOGGER_NAME)


class GPUDeviceFactory:
    """
    Factory class for creating GPUDevices
    """

    def __init__(self):
        self._devices = []
        self._devices_by_bus_id = {}
        self._devices_by_uuid = {}
        self.init_all_devices()

    def init_all_devices(self, dcgmPath=None):
        """
        Create GPUDevice objects for all DCGM visible
        devices.

        Parameters
        ----------
        dcgmPath : str
            Absolute path to dcgm shared library
        """

        if numba.cuda.is_available():
            logger.info("Initiliazing GPUDevice handles")
            structs._dcgmInit(dcgmPath)
            dcgm_agent.dcgmInit()

            # Start DCGM in the embedded mode to use the shared library
            dcgm_handle = dcgm_agent.dcgmStartEmbedded(
                structs.DCGM_OPERATION_MODE_MANUAL)

            # Create a GPU device for every supported DCGM device
            dcgm_device_ids = dcgm_agent.dcgmGetAllSupportedDevices(dcgm_handle)

            for device_id in dcgm_device_ids:
                device_atrributes = dcgm_agent.dcgmGetDeviceAttributes(
                    dcgm_handle, device_id).identifiers
                pci_bus_id = device_atrributes.pciBusId.upper()
                device_uuid = device_atrributes.uuid
                device_name = device_atrributes.deviceName
                gpu_device = GPUDevice(device_name, device_id, pci_bus_id,
                                       device_uuid)

                self._devices.append(gpu_device)
                self._devices_by_bus_id[pci_bus_id] = gpu_device
                self._devices_by_uuid[device_uuid] = gpu_device

            dcgm_agent.dcgmShutdown()

    def get_device_by_bus_id(self, bus_id, dcgmPath=None):
        """
        Get a GPU device by using its bus ID.

        Parameters
        ----------
        bus_id : bytes
            Bus id corresponding to the GPU. The bus id should be created by
            converting the colon separated hex notation into a bytes type
            using ascii encoding. The bus id before conversion to bytes
            should look like "00:65:00".

        Returns
        -------
        Device
            The device associated with this bus id.
        """

        if bus_id in self._devices_by_bus_id:
            return self._devices_by_bus_id[bus_id]
        else:
            raise TorchBenchAnalyzerException(
                f'GPU with {bus_id} bus id is either not supported by DCGM or not present.'
            )

    def get_device_by_cuda_index(self, index):
        """
        Get a GPU device using the CUDA index. This includes the index
        provided by CUDA visible devices.

        Parameters
        ----------
        index : int
            index of the device in the list of visible CUDA devices.

        Returns
        -------
        Device
            The device associated with the index provided.

        Raises
        ------
        IndexError
            If the index is out of bound.
        """

        devices = numba.cuda.list_devices()
        if index > len(devices) - 1:
            raise IndexError

        cuda_device = devices[index]
        device_identity = cuda_device.get_device_identity()
        pci_domain_id = device_identity['pci_domain_id']
        pci_device_id = device_identity['pci_device_id']
        pci_bus_id = device_identity['pci_bus_id']
        device_bus_id = \
            f'{pci_domain_id:08X}:{pci_bus_id:02X}:{pci_device_id:02X}.0'

        return self.get_device_by_bus_id(device_bus_id)

    def get_device_by_uuid(self, uuid, dcgmPath=None):
        """
        Get a GPU device using the GPU uuid.

        Parameters
        ----------
        uuid : str
            index of the device in the list of visible CUDA devices.

        Returns
        -------
        Device
            The device associated with the uuid.

        Raises
        ------
        TritonModelAnalyzerExcpetion
            If the uuid does not exist this exception will be raised.
        """

        if uuid in self._devices_by_uuid:
            return self._devices_by_uuid[uuid]
        else:
            raise TorchBenchAnalyzerException(
                f'GPU UUID {uuid} was not found.')

    def verify_requested_gpus(self, requested_gpus):
        """
        Creates a list of GPU UUIDs corresponding to the GPUs visible to
        numba.cuda among the requested gpus

        Parameters
        ----------
        requested_gpus : list of str or list of ints
            Can either be GPU UUIDs or GPU device ids

        Returns
        -------
        List of GPUDevices
            list of GPUDevices corresponding to visible GPUs among requested
        
        Raises
        ------
        TorchBenchAnalyzerException
        """

        cuda_visible_gpus = self.get_cuda_visible_gpus()

        if len(requested_gpus) == 1:
            if requested_gpus[0] == 'all':
                self._log_gpus_used(cuda_visible_gpus)
                return cuda_visible_gpus
            elif requested_gpus[0] == '[]':
                logger.info("No GPUs requested")
                return []

        try:
            # Check if each string in the list can be parsed as an int
            requested_cuda_indices = list(map(int, requested_gpus))
            requested_gpus = []

            for idx in requested_cuda_indices:
                try:
                    requested_gpus.append(self.get_device_by_cuda_index(idx))
                except TorchBenchAnalyzerException:
                    raise TorchBenchAnalyzerException(
                        f"Requested GPU with device id : {idx}. This GPU is not supported by DCGM."
                    )
        except ValueError:
            # requested_gpus are assumed to be UUIDs
            requested_gpus = [
                self.get_device_by_uuid(uuid) for uuid in requested_gpus
            ]
            pass

        # Return the intersection of CUDA visible UUIDs and requested/supported UUIDs.
        available_gpus = list(set(cuda_visible_gpus) & set(requested_gpus))
        self._log_gpus_used(available_gpus)

        return available_gpus

    def get_cuda_visible_gpus(self):
        """
        Returns
        -------
        list of GPUDevice
            UUIDs of the DCGM supported devices visible to CUDA
        """

        cuda_visible_gpus = []
        if numba.cuda.is_available():
            for cuda_device in numba.cuda.list_devices():
                try:
                    cuda_visible_gpus.append(
                        self.get_device_by_cuda_index(cuda_device.id))
                except TorchBenchAnalyzerException:
                    # Device not supported by DCGM, log warning
                    logger.warning(
                        f"Device '{str(cuda_device.name, encoding='ascii')}' with "
                        f"cuda device id {cuda_device.id} is not supported by DCGM."
                    )
        return cuda_visible_gpus

    def _log_gpus_used(self, gpus):
        """
        Log the info for the GPUDevices in use
        """

        for gpu in gpus:
            logger.info(
                f"Using GPU {gpu.device_id()} {gpu.device_name()} with UUID {gpu.device_uuid()}"
            )
