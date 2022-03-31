import os
import subprocess
import sys
import signal
import textwrap
import typing

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

try:
    from components._impl.workers import base as base_worker
    from components._impl.workers import in_process_worker
    from components._impl.workers import subprocess_worker
    from components._impl.workers import subprocess_rpc
except (ImportError, ModuleNotFoundError):
    print(f"""
        This test must be run from the repo root directory as
        `python -m components.test.{os.path.splitext(os.path.basename(__file__))[0]}`
    """)
    raise


class CustomClass:
    """Used to test handline of non-builtin objects."""
    pass


class TestBenchmarkWorker(TestCase):

    def _test_namespace_isolation(self, worker: base_worker.WorkerBase):
        worker_global_vars: typing.Dict[str, str] = worker.load_stmt(
            r"{k: repr(type(v)) for k, v in globals().items()}")
        allowed_keys = {
            "__builtins__",
            subprocess_rpc.WORKER_IMPL_NAMESPACE,
        }
        extra_vars = {
            k: v for k, v in worker_global_vars.items()
            if k not in allowed_keys
        }
        self.assertDictEqual(extra_vars, {})

    def _subtest_cleanup(
        self,
        worker: base_worker.WorkerBase,
        test_vars: typing.Tuple[str, ...]
    ) -> None:
        worker.run("\n".join([f"del {v}" for v in test_vars]))
        self._test_namespace_isolation(worker)

    def _check_basic_store_and_load(self, worker: base_worker.WorkerBase) -> None:
        worker.store("y", 2)
        self.assertEqual(worker.load("y"), 2)

        worker.run("del y")

        with self.assertRaisesRegex(NameError, "name 'y' is not defined"):
            worker.load("y")

    def _check_custom_store_and_load(self, worker: base_worker.WorkerBase) -> None:
        with self.assertRaisesRegex(ValueError, "unmarshallable object"):
            worker.store("my_class", CustomClass())

        worker.run("""
            class CustomClass:
                pass

            my_class = CustomClass()
        """)
        with self.assertRaisesRegex(ValueError, "unmarshallable object"):
            worker.load("my_class")

        self._subtest_cleanup(worker, ("my_class", "CustomClass"))

    def _check_load_stmt(self, worker: base_worker.WorkerBase) -> None:
        self.assertDictEqual(
            {"a": 1 + 3, 2: "b"},
            worker.load_stmt('{"a": 1 + 3, 2: "b"}'),
        )
        self._subtest_cleanup(worker, ())

    def _check_complex_stmts(self, worker: base_worker.WorkerBase) -> None:
        worker.run("""
            def test_fn():
                x = 10
                y = 2

                # Make sure we can handle blank lines.
                return x + y
            z = test_fn()
        """)
        self.assertEqual(worker.load("z"), 12)

        # Ensure variables persist across invocations. (In this case, `f`)
        worker.run("z = test_fn() + 1")
        self.assertEqual(worker.load("z"), 13)

        # Ensure invocations have access to global variables.
        worker.store("captured_var", 5)
        worker.run("""
            def test_fn():
                # Make sure closures work properly
                return captured_var + 1
            z = test_fn()
        """)
        self.assertEqual(worker.load("z"), 6)

        self._subtest_cleanup(worker, ("captured_var", "z", "test_fn"))

    def _check_environment_consistency(self, worker: base_worker.WorkerBase) -> None:
        # It is important that the worker mirrors the caller. Otherwise imports
        # may not resolve, or may resolve to incorrect paths. As a result, the
        # worker must ensure that it faithfully reproduces the caller's
        # environment.
        worker.run("""
            import os
            import sys

            cwd = os.getcwd()
            sys_executable = sys.executable
            sys_path = sys.path
        """)
        self.assertEqual(worker.load("cwd"), os.getcwd())
        self.assertEqual(worker.load("sys_executable"), sys.executable)
        self.assertEqual(worker.load("sys_path"), sys.path)

        # Environment parity is especially important for `torch`, since
        # importing an incorrect version will result in silently garbage
        # results.
        worker.run("""
            import torch
            torch_file = torch.__file__
        """)
        self.assertEqual(worker.load("torch_file"), torch.__file__)

        self._subtest_cleanup(
            worker,
            ("os", "sys", "cwd", "sys_executable", "sys_path", "torch", "torch_file"),
        )

    def _test_exceptions(self, worker: base_worker.WorkerBase):
        with self.assertRaisesRegex(AssertionError, "False is not True"):
            worker.run("assert False, 'False is not True'")

        with self.assertRaisesRegex(ValueError, "Test msg"):
            worker.run("raise ValueError('Test msg')")

    def _test_child_trace_exception(
        self,
        worker: subprocess_worker.SubprocessWorker,
    ) -> None:
        try:
            worker.run("print('This should not appear.')")
            worker.run("""
                print("This is not going to work")
                with open("this_file_does_not_exist") as f:
                    pass
            """)
            self.fail("Worker should have raised.")

        except FileNotFoundError as e:
            e_cause = e.__cause__
            self.assertIsInstance(e_cause, subprocess_rpc.ChildTraceException)
            extra_debug_info: str = e_cause.args[0]
            assert isinstance(extra_debug_info, str)

            # stdout / stderr plumbing is only for the failing snippet. Print
            # stmts from earlier expressions should not be included.
            self.assertNotRegex(extra_debug_info, "This should not appear")

            # Make sure the worker provided a stack trace.
            self.assertRegex(
                extra_debug_info,
                textwrap.dedent(r"""
                Traceback \(most recent call last\):
                \s+ File.*subprocess_rpc.*
                """).strip()
            )

            self.assertRegex(
                extra_debug_info,
                r"No such file or directory: .this_file_does_not_exist."
            )

            # Make sure stdout / stderr were plumbed from the worker.
            self.assertRegex(extra_debug_info, "This is not going to work")

    def _generic_worker_tests(self, worker: base_worker.WorkerBase) -> None:
        # Make sure we have a clean start.
        self._test_namespace_isolation(worker)

        self._check_basic_store_and_load(worker)
        self._check_load_stmt(worker)
        self._check_custom_store_and_load(worker)
        self._check_complex_stmts(worker)
        self._check_environment_consistency(worker)
        self._test_exceptions(worker)

        self._test_namespace_isolation(worker)

    def test_in_process_worker(self) -> None:
        worker = in_process_worker.InProcessWorker(globals={})
        self._generic_worker_tests(worker)

        # InProcessWorker specific tests include passing non-empty globals.
        worker = in_process_worker.InProcessWorker(globals={"x": 1})

        # Make sure worker is actually using globals passed.
        self.assertEqual(worker.load("x"), 1)

        # Test `in_memory` exception for InProcessWorker.
        worker.store("my_class", CustomClass(), in_memory=True)
        self.assertIsInstance(worker._globals["my_class"], CustomClass)

    def test_subprocess_worker(self) -> None:
        worker = subprocess_worker.SubprocessWorker()
        self._generic_worker_tests(worker)
        self._test_child_trace_exception(worker)

    def test_subprocess_worker_segv_handling(self):
        worker = subprocess_worker.SubprocessWorker(timeout=1)
        with self.assertRaisesRegex(OSError, f"Subprocess terminates with code {int(signal.SIGSEGV)}"):
            worker.run("""
                import os
                import signal
                os.kill(os.getpid(), signal.SIGSEGV)
            """)

    def test_subprocess_worker_fault_handling(self):
        worker = subprocess_worker.SubprocessWorker(timeout=1)
        with self.assertRaisesRegex(OSError, "Exceeded timeout"):
            worker.run("""
                import os
                os._exit(0)
            """)

        self.assertFalse(worker.alive)
        with self.assertRaisesRegex(AssertionError, "Process has exited"):
            worker.run("pass")

        worker = subprocess_worker.SubprocessWorker(timeout=1)
        with self.assertRaisesRegex(OSError, "Exceeded timeout"):
            worker.run("""
                import time
                time.sleep(2)
            """)

        # Once a command times out, the integrity of the underlying subprocess
        # cannot be guaranteed and the worker neeeds to refuse future work.
        # This is different from an Exception being thrown in the subprocess;
        # in that case communication is still in an orderly state.
        self.assertFalse(worker.alive)
        with self.assertRaisesRegex(AssertionError, "Process has exited"):
            worker.run("pass")

        # Make sure `_kill_proc` is idempotent.
        worker._kill_proc()
        worker._kill_proc()

    def test_subprocess_worker_sys_exit(self):
        worker = subprocess_worker.SubprocessWorker(timeout=1)
        with self.assertRaisesRegex(subprocess_rpc.UnserializableException, "SystemExit"):
            worker.run("import sys")
            worker.run("sys.exit()")


if __name__ == '__main__':
    run_tests()
