from .record import Record


class CPURecord(Record):
    """
    This is a base class for any
    CPU based record
    """

    def __init__(self, value, timestamp=0):
        """
        Parameters
        ----------
        value : float
            The value of the CPU metrtic
        device_uuid : str
            A dummy parameter to pass record aggregator.
        timestamp : int
            The timestamp for the record in nanoseconds
        """

        super().__init__(value, timestamp)
        self._device_uuid = 0x1

    def device_uuid(self):
        return self._device_uuid
