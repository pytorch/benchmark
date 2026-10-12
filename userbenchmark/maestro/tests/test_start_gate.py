"""Regressions for event ordering, host release, and GPU launch gating."""

import ast
import ctypes
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "maestro"


def load_gate(torch):
    spec = importlib.util.spec_from_file_location("_start_gate_test", SOURCE / "core/utils/start_gate.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch}):
        spec.loader.exec_module(module)
    return module.CUDAStartGate


class StartGateTest(unittest.TestCase):
    def setUp(self):
        self.flag = ctypes.c_uint32(1)
        self.log = []
        stream = types.SimpleNamespace(cuda_stream=7, synchronize=Mock())
        self.stream = stream

        class Event:
            recorded = False

            def record(event, recorded_stream):
                self.assertIs(recorded_stream, stream)
                self.assertEqual(self.flag.value, 0)
                self.log.append("record")
                event.recorded = True

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(Stream=lambda: stream, Event=Event,
                                          current_stream=lambda: types.SimpleNamespace(wait_stream=Mock()))
        self.torch = torch
        driver = types.SimpleNamespace(
            CU_MEMHOSTALLOC_DEVICEMAP=2,
            CUstreamWaitValue_flags=types.SimpleNamespace(CU_STREAM_WAIT_VALUE_EQ=1),
            cuMemHostAlloc=Mock(return_value=(0, ctypes.addressof(self.flag))),
            cuMemHostGetDevicePointer=Mock(return_value=(0, 99)),
            cuMemFreeHost=Mock(return_value=(0,)),
            cuStreamWaitValue32=Mock(side_effect=lambda *args: (self.log.append("wait") or 0,)),
        )
        self.driver = driver
        bindings = types.ModuleType("cuda.bindings")
        bindings.driver = driver
        self.modules = patch.dict(sys.modules, {"cuda": types.ModuleType("cuda"), "cuda.bindings": bindings})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.gate = load_gate(torch)()
        self.addCleanup(self.gate.close)

    def axis(self):
        def wait_event(event):
            self.assertTrue(event.recorded)
            self.assertEqual(self.flag.value, 0)
            self.log.append("axis_wait")
        return types.SimpleNamespace(wait_event=wait_event)

    def test_recorded_event_holds_all_axes_until_host_release(self):
        with self.gate.hold([self.axis(), self.axis()]):
            self.assertEqual(self.log, ["wait", "record", "axis_wait", "axis_wait"])
            self.assertEqual(self.flag.value, 0)
            self.driver.cuStreamWaitValue32.assert_called_once_with(7, 99, 1, 1)
        self.assertEqual(self.flag.value, 1)

    def test_enqueue_exception_releases_gate(self):
        with self.assertRaisesRegex(ValueError, "enqueue failed"):
            with self.gate.hold([self.axis()]):
                raise ValueError("enqueue failed")
        self.assertEqual(self.flag.value, 1)

    def test_cuda_error_releases_gate(self):
        self.driver.cuStreamWaitValue32.side_effect = None
        self.driver.cuStreamWaitValue32.return_value = (801,)
        with self.assertRaisesRegex(RuntimeError, "CUDA start gate failed"):
            with self.gate.hold([self.axis()]):
                self.fail("CUDA error must prevent enqueue")
        self.assertEqual(self.flag.value, 1)

    def test_close_synchronizes_and_frees_once(self):
        self.gate.close()
        self.gate.close()
        self.stream.synchronize.assert_called_once()
        self.driver.cuMemFreeHost.assert_called_once_with(ctypes.addressof(self.flag))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with self.gate.hold([]):
                pass

    def test_execute_enqueues_all_iterations_before_release(self):
        tree = ast.parse((SOURCE / "core/orchestrator.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Orchestrator")
        execute = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_execute")
        namespace = {"torch": self.torch}
        exec(compile(ast.Module(body=[execute], type_ignores=[]), "_execute", "exec"), namespace)
        axis = Mock()
        axis.wait_event.side_effect = self.axis().wait_event
        axis.use_axis.side_effect = lambda: nullcontext()
        block = types.SimpleNamespace(axis=axis, run=lambda: self.log.append(("block", self.flag.value)))
        def barrier(async_op=False):
            self.log.append(("barrier", async_op, self.flag.value))
            return Mock() if async_op else None
        obj = types.SimpleNamespace(
            _start_gate=self.gate, oob_dist_utils=types.SimpleNamespace(barrier=barrier),
            axis_cls=types.SimpleNamespace(synchronize_all=lambda: self.log.append(("sync", self.flag.value))),
        )
        namespace["_execute"](obj, [block], iters=2)
        self.assertEqual(self.log[0], ("barrier", False, 1))
        self.assertEqual(self.log.count(("block", 0)), 2)
        self.assertEqual(self.log.count(("barrier", True, 0)), 2)
        self.assertEqual(self.log[-1], ("sync", 1))


if __name__ == "__main__":
    unittest.main()
