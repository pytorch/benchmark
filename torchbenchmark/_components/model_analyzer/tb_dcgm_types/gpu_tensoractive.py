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

from functools import total_ordering

from .gpu_record import GPURecord


@total_ordering
class GPUTensorActive(GPURecord):
    """
    GPU Tensor active record
    """

    tag = "gpu_tensoractive"

    def __init__(self, value, device_uuid=None, timestamp=0):
        """
        Parameters
        ----------
        value : float
            The value of the GPU metrtic
        device_uuid : str
            The  GPU device uuid this metric is associated
            with.
        timestamp : int
            The timestamp for the record in nanoseconds
        """

        super().__init__(value, device_uuid, timestamp)

    @staticmethod
    def aggregation_function():
        """
        The function that is used to aggregate
        this type of record
        """

        def average(seq):
            return sum(seq[1:], start=seq[0]) / len(seq)

        return average

    @staticmethod
    def header(aggregation_tag=False):
        """
        Parameters
        ----------
        aggregation_tag: bool
            An optional tag that may be displayed
            as part of the header indicating that
            this record has been aggregated using
            max, min or average etc.

        Returns
        -------
        str
            The full name of the
            metric.
        """

        return ("Average " if aggregation_tag else "") + "GPU FP32 Active (%)"

    def __eq__(self, other):
        """
        Allows checking for
        equality between two records
        """

        return self.value() == other.value()

    def __lt__(self, other):
        """
        Allows checking if
        this record is less than
        the other
        """

        return self.value() < other.value()

    def __add__(self, other):
        """
        Allows adding two records together
        to produce a brand new record.
        """

        return GPUTensorActive(device_uuid=None, value=(self.value() + other.value()))

    def __sub__(self, other):
        """
        Allows subtracting two records together
        to produce a brand new record.
        """

        return GPUTensorActive(device_uuid=None, value=(self.value() - other.value()))
