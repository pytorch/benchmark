from functools import total_ordering
from .cpu_record import CPURecord


@total_ordering
class CPUPeakMemory(CPURecord):
    """
    The peak memory usage in the CPU.
    """

    tag = "cpu_peak_memory"

    def __init__(self, value, timestamp=0):
        """
        Parameters
        ----------
        value : float
            The value of the CPU metrtic
        timestamp : int
            The timestamp for the record in nanoseconds
        """

        super().__init__(value, timestamp)
        

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

        return ("Max " if aggregation_tag else "") + "GPU Memory Usage (MB)"

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

        return self.value() > other.value()

    def __add__(self, other):
        """
        Allows adding two records together
        to produce a brand new record.
        """

        return CPUPeakMemory(value=(self.value() + other.value()))

    def __sub__(self, other):
        """
        Allows subtracting two records together
        to produce a brand new record.
        """

        return CPUPeakMemory(value=(other.value() - self.value()))
