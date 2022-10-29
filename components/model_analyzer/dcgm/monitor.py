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

from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
import time

from ..tb_dcgm_types.da_exceptions import TorchBenchAnalyzerException



class Monitor(ABC):
    """
    Monitor abstract class is a parent class used for monitoring devices.
    """

    def __init__(self, frequency, metrics):
        """
        Parameters
        ----------
        frequency : float
            How often the metrics should be monitored. It is in seconds.
        metrics : list
            A list of Record objects that will be monitored.

        Raises
        ------
        TorchBenchAnalyzerException
        """

        self._frequency = frequency

        # Is the background thread active
        self._thread_active = False

        # Background thread collecting results
        self._thread = None

        # Thread pool
        self._thread_pool = ThreadPool(processes=1)
        self._metrics = metrics

    def _monitoring_loop(self):
        frequency = self._frequency

        while self._thread_active:
            begin = time.time()
            # Monitoring iteration implemented by each of the subclasses
            self._monitoring_iteration()
            # print("======working======")
            duration = time.time() - begin
            if duration < frequency:
                time.sleep(frequency - duration)

    @abstractmethod
    def _monitoring_iteration(self):
        """
        Each of the subclasses must implement this.
        This is called to execute a single round of monitoring.
        """

        pass

    @abstractmethod
    def _collect_records(self):
        """
        This method is called to collect all the monitoring records.
        It is called in the stop_recording_metrics function after
        the background thread has stopped.

        Returns
        -------
        List of Records
            The list of records collected by the monitor
        """

        pass

    def start_recording_metrics(self):
        """
        Start recording the metrics.
        """

        self._thread_active = True
        self._thread = self._thread_pool.apply_async(self._monitoring_loop)

    def stop_recording_metrics(self):
        """
        Stop recording metrics. This will stop monitring all the metrics.

        Returns
        ------
        List of Records

        Raises
        ------
        TorchBenchAnalyzerException
        """

        if not self._thread_active:
            raise TorchBenchAnalyzerException(
                "start_recording_metrics should be "
                "called before stop_recording_metrics")

        self._thread_active = False
        self._thread = None

        return self._collect_records()

    def destroy(self):
        """
        Cleanup threadpool resources
        """

        self._thread_pool.terminate()
        self._thread_pool.close()
