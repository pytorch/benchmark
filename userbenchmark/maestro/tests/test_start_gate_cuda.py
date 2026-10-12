"""CUDA regression for a gate which remains closed until host release."""

import importlib.util
from pathlib import Path
import unittest


class CUDAStartGateTest(unittest.TestCase):
    def test_axis_work_stays_pending_until_release(self):
        try:
            import torch
            from cuda.bindings import driver
        except ImportError:
            self.skipTest("PyTorch and cuda-bindings are required")
        if not torch.cuda.is_available():
            self.skipTest("CUDA GPU required")

        source = Path(__file__).resolve().parents[1] / "maestro/core/utils/start_gate.py"
        spec = importlib.util.spec_from_file_location("_cuda_start_gate_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for device in range(torch.cuda.device_count()):
            with self.subTest(device=device), torch.cuda.device(device):
                gate = module.CUDAStartGate()
                axes = [torch.cuda.Stream(), torch.cuda.Stream()]
                # Preallocate and initialize before the gate; allocations can sync.
                outputs = [torch.zeros(16, device="cuda") for _ in axes]
                torch.cuda.synchronize()
                try:
                    for value in (3, 5):
                        done = [torch.cuda.Event() for _ in axes]
                        with gate.hold(axes):
                            for axis, output, event in zip(axes, outputs, done):
                                with torch.cuda.stream(axis):
                                    output.fill_(value)
                                    event.record()
                            self.assertFalse(any(event.query() for event in done))
                        for axis in axes:
                            axis.synchronize()
                        self.assertTrue(all(event.query() for event in done))
                        self.assertTrue(all(torch.all(output == value).item() for output in outputs))
                finally:
                    gate.close()


if __name__ == "__main__":
    unittest.main()
