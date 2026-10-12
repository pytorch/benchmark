"""Hold GPU launches until the host has enqueued all axes' work."""

import ctypes
from contextlib import contextmanager

import torch


class CUDAStartGate:
    """A recorded event behind a host-released CUDA stream memory wait.

    The flag lives in mapped pinned host memory. Releasing it on the CPU avoids
    depending on another GPU stream which might itself be waiting for the gate.
    Axes depend on the recorded event, giving CUDA explicit stream dependencies.
    No kernel, memcpy, or memset is launched, so CUPTI's workload capture does
    not include gate activities. Calls to hold() must be separated by GPU sync.
    """

    def __init__(self):
        from cuda.bindings import driver

        self._driver = driver
        self._stream = torch.cuda.Stream()
        self._host_ptr = self._check(
            driver.cuMemHostAlloc(ctypes.sizeof(ctypes.c_uint32), driver.CU_MEMHOSTALLOC_DEVICEMAP)
        )
        try:
            self._device_ptr = self._check(driver.cuMemHostGetDevicePointer(self._host_ptr, 0))
            self._flag = ctypes.c_uint32.from_address(int(self._host_ptr))
            self._flag.value = 1
        except BaseException:
            self._check(driver.cuMemFreeHost(self._host_ptr))
            self._host_ptr = None
            raise

    @staticmethod
    def _check(result):
        status, *values = result
        if int(status) != 0:
            raise RuntimeError(f"CUDA start gate failed: {status}")
        return values[0] if values else None

    @contextmanager
    def hold(self, axes):
        if self._host_ptr is None:
            raise RuntimeError("CUDA start gate is closed")
        self._flag.value = 0
        try:
            self._check(self._driver.cuStreamWaitValue32(
                self._stream.cuda_stream, self._device_ptr, 1,
                self._driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
            ))
            event = torch.cuda.Event()
            # Record BEFORE any axis waits: an unrecorded CUDA event cannot gate.
            event.record(self._stream)
            for axis in axes:
                axis.wait_event(event)
            yield
        finally:
            # Also unblock queued work when host-side enqueue raises.
            self._flag.value = 1

    def close(self):
        if self._host_ptr is not None:
            self._flag.value = 1
            self._stream.synchronize()
            self._check(self._driver.cuMemFreeHost(self._host_ptr))
            self._host_ptr = None
