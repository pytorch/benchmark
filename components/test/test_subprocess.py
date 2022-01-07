"""Unit tests specifically for the components of SubprocessWorker.

End-to-end tests (e.g. does SubprocessWorker properly implement the
WorkerBase API) still live in `test_worker`.
"""

import functools
import os
import sys
import textwrap
import threading
import typing

from torch.testing._internal.common_utils import TestCase, run_tests

try:
    from components._impl.tasks import base as task_base
    from components._impl.workers import subprocess_rpc
except (ImportError, ModuleNotFoundError):
    print(f"""
        This test must be run from the repo root directory as
        `python -m components.test.{os.path.splitext(os.path.basename(__file__))[0]}`
    """)
    raise


class TestParseFunction(TestCase):

    @staticmethod
    def _indent(s: str) -> str:
        return textwrap.indent(s, " " * 12)

    def test_parse_trivial(self) -> None:
        def f(x: int) -> None:
            pass

        _, body = task_base.parse_f(f)
        self.assertExpectedInline(
            self._indent(body), """\
            pass""",
        )

    def test_parse_simple(self) -> None:
        def f(
            x: int,
        ) -> None:
            for _ in range(10):
                pass

        _, body = task_base.parse_f(f)
        self.assertExpectedInline(
            self._indent(body), """\
            for _ in range(10):
                pass""",
        )

    def test_parse_inline(self) -> None:
        def f(x: typing.Any, y: int = 1) -> None: print([x for _ in range(y)])

        _, body = task_base.parse_f(f)
        self.assertExpectedInline(
            self._indent(body), """\
            print([x for _ in range(y)])""",
        )

    def test_parse_with_comments(self) -> None:
        def f(
            x: int,   # This is a comment
            y: bool,  # also a comment

            # More comments.
        ) -> typing.Any:  # Comment on return line.
            """Docstring

            Note: This will be dropped in Python 3.7. See `parse_f` for details.
            """

            x += 1

            y = """
                This is preserved.
            """

            # Comment in src.
            return y

        _, body = task_base.parse_f(f)
        # Python 3.7 removes docstring but 3.8+ doesn't. See `parse_f` for details.
        docstring = """\
            \"\"\"Docstring

            Note: This will be dropped in Python 3.7. See `parse_f` for details.
            \"\"\"\n\n""" if not sys.version_info[:2] == (3,7) else ""
        self.assertExpectedInline(
            self._indent(body), f"""{docstring}\
            x += 1

            y = \"\"\"
                This is preserved.
            \"\"\"

            # Comment in src.
            return y""",
            )

    def test_parse_method(self) -> None:
        class MyClass:

            @staticmethod
            def f(x: int) -> int:
                """Identity, but with more steps"""
                return x

            @staticmethod
            def g(x: int) -> int:
                """Identity, but with more steps

                Culled, as this is a multi-line docstring
                """
                return x

        _, body = task_base.parse_f(MyClass.f)
        self.assertExpectedInline(
            self._indent(body), """\
            \"\"\"Identity, but with more steps\"\"\"
            return x""",
        )

        _, body = task_base.parse_f(MyClass.g)
        # Python 3.7 removes docstring but 3.8+ doesn't. See `parse_f` for details.
        docstring = """\
            \"\"\"Identity, but with more steps

            Culled, as this is a multi-line docstring
            \"\"\"\n""" if not sys.version_info[:2] == (3, 7) else ""
        self.assertExpectedInline(
            self._indent(body), f"""{docstring}\
            return x""",
        )

    def test_parse_pathological(self) -> None:
        def f(
            x: \
            int,
            y: typing.Dict[str, int],
            *,
            z: str,
        # Isn't that a charming (but legal) indentation?
        ) \
        -> typing.Optional[typing.Union[
        float, int]
                    ]:  # Just for good measure.
            """Begin the actual body.

            (For better or worse...)
            """
            del x
            q = y.get(
                z,
                None,
            )

            # Intermediate comment

            if False:
                return 1
            elif q:
                raise ValueError

            q = 1

        _, body = task_base.parse_f(f)
        # Python 3.7 removes docstring but 3.8+ doesn't. See `parse_f` for details.
        docstring = """\
            \"\"\"Begin the actual body.

            (For better or worse...)
            \"\"\"\n""" if not sys.version_info[:2]==(3,7) else ""
        self.assertExpectedInline(
            self._indent(body), f"""{docstring}\
            del x
            q = y.get(
                z,
                None,
            )

            # Intermediate comment

            if False:
                return 1
            elif q:
                raise ValueError

            q = 1""",
        )

    def test_fully_typed(self) -> None:
        def f(x):
            pass

        with self.assertRaisesRegex(
            TypeError,
            "Missing type annotation for parameter `x`"
        ):
            task_base.parse_f(f)

        def g(x: int):
            pass

        with self.assertRaisesRegex(
            TypeError,
            "Missing return annotation."
        ):
            task_base.parse_f(g)

    def test_no_functor(self) -> None:
        class F:

            def __call__(self) -> None:
                pass

        with self.assertRaisesRegex(TypeError, "Expected function, got"):
            task_base.parse_f(F())

    def test_no_variadic(self) -> None:
        def f(*args) -> None:
            pass

        with self.assertRaisesRegex(
            TypeError,
            r"Variadic positional argument `\*args` not permitted for `run_in_worker` function."
        ):
            task_base.parse_f(f)

        def g(**kwargs) -> None:
            pass

        with self.assertRaisesRegex(
            TypeError,
            r"Variadic keywork argument `\*\*kwargs` not permitted for `run_in_worker` function."
        ):
            task_base.parse_f(g)

    def test_no_decorator(self) -> None:

        def my_decorator(f: typing.Callable) -> typing.Callable:

            @functools.wraps(f)
            def g(*args, **kwargs) -> typing.Any:
                return f(*args, **kwargs)

            return g

        @my_decorator
        def f() -> None:
            pass

        with self.assertRaisesRegex(
            TypeError,
            "`f` cannot be decorated below `@run_in_worker`"
        ):
            task_base.parse_f(f)


class TestSubprocessRPC(TestCase):

    def test_pipe_basic_read_write(self) -> None:
        pipe = subprocess_rpc.Pipe()

        # Test small read.
        msg = b"abc"
        pipe.write(msg)
        self.assertEqual(msg, pipe.read())

        # Test large read.
        msg = b"asdjkf" * 1024
        pipe.write(msg)
        self.assertEqual(msg, pipe.read())

    def test_pipe_stacked_read_write(self) -> None:
        pipe = subprocess_rpc.Pipe()

        pipe.write(b"abc")
        pipe.write(b"def")
        pipe.write(b"ghi")
        self.assertEqual(b"abc", pipe.read())
        self.assertEqual(b"def", pipe.read())
        self.assertEqual(b"ghi", pipe.read())

    def test_pipe_clone(self) -> None:
        msg = b"msg"
        pipe = subprocess_rpc.Pipe()
        alt_pipe_0 = subprocess_rpc.Pipe(write_handle=pipe.write_handle)
        alt_pipe_0.write(msg)
        self.assertEqual(msg, pipe.read())
        with self.assertRaises(IOError):
            alt_pipe_0.read()

        alt_pipe_1 = subprocess_rpc.Pipe(read_handle=pipe.read_handle)
        pipe.write(msg)
        self.assertEqual(msg, alt_pipe_1.read())
        with self.assertRaises(IOError):
            alt_pipe_1.write(msg)

    def test_pipe_timeout(self) -> None:
        result = {}
        def callback():
            result["callback_run"] = True

        # We have to run this in a thread, because if the timeout mechanism
        # fails we don't want the entire unit test suite to hang.
        pipe = subprocess_rpc.Pipe(timeout=0.5, timeout_callback=callback)
        def target():
            try:
                pipe.read()
            except Exception as e:
                result["e"] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=10)
        e: typing.Optional[Exception] = result.get("e", None)
        self.assertIsNotNone(e)
        with self.assertRaisesRegex(OSError, "Exceeded timeout: 0.5"):
            raise e

        self.assertTrue(result.get("callback_run", None), True)

    def test_pipe_concurrent_timeout(self) -> None:
        result = {"callback_count": 0, "exceptions": []}
        def callback():
            result["callback_count"] += 1

        timeouts = [0.5, 1.0, 1.5]
        pipes = [
            subprocess_rpc.Pipe(timeout=timeout, timeout_callback=callback)
            for timeout in timeouts
        ]

        def target(pipe):
            try:
                pipe.read()
            except Exception as e:
                result["exceptions"].append(e)

        threads = [threading.Thread(target=target, args=(pipe,)) for pipe in pipes]
        [t.start() for t in threads]
        [t.join(timeout=5) for t in threads]
        self.assertEqual(result["callback_count"], 3)
        self.assertEqual(len(result["exceptions"]), 3)
        for e in result["exceptions"]:
            with self.assertRaisesRegex(OSError, "Exceeded timeout:"):
                raise e

    def test_pipe_cleanup(self) -> None:
        assertTrue = self.assertTrue
        assertFalse = self.assertFalse
        del_audit = {"count": 0}

        class OwnCheckingPipe(subprocess_rpc.Pipe):

            def __init__(self):
                super().__init__()
                self._cleanup_was_run = False
                assertTrue(self._owns_pipe)

            def _close_fds(self) -> None:
                super()._close_fds()
                self._cleanup_was_run = True

            def __del__(self) -> None:
                super().__del__()
                assert self._cleanup_was_run
                del_audit["count"] += 1

        class NonOwnCheckingPipe(subprocess_rpc.Pipe):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assertFalse(self._owns_pipe)

            def _close_fds(self) -> None:
                raise IOError("This would damage the owning pipe")

            def __del__(self) -> None:
                super().__del__()
                del_audit["count"] += 1

        pipe = OwnCheckingPipe()
        alt_pipe_0 = NonOwnCheckingPipe(read_handle=pipe.read_handle)
        alt_pipe_1 = NonOwnCheckingPipe(write_handle=pipe.write_handle)
        alt_pipe_2 = NonOwnCheckingPipe(
            read_handle=pipe.read_handle,
            write_handle=pipe.write_handle,
        )

        del pipe
        del alt_pipe_0
        del alt_pipe_1
        del alt_pipe_2

        # Make sure the tests we expect in __del__ actually ran.
        self.assertEqual(del_audit["count"], 4)


class TestSubprocessExceptions(TestCase):

    def _test_raise(
        self,
        raise_type: typing.Type[Exception],
        reraise_type: typing.Type[Exception],
    ) -> None:
        try:
            raise raise_type("Fail")
        except Exception as e:
            e_raised = e  # `e` is scoped to the `except` block
            tb = sys.exc_info()[2]
            serialized_e = subprocess_rpc.SerializedException.from_exception(e=e, tb=tb)

        with self.assertRaises(reraise_type):
            subprocess_rpc.SerializedException.raise_from(serialized_e)

        if raise_type is reraise_type:
            try:
                subprocess_rpc.SerializedException.raise_from(serialized_e)
                self.fail("`raise_from` failed to raise.")
            except Exception as e:
                self.assertEqual(e_raised.args, e.args)

    def _test_raise_builtin(self, raise_type: typing.Type[Exception]) -> None:
        self._test_raise(raise_type=raise_type, reraise_type=raise_type)

    def test_unserializable(self) -> None:
        # Make sure we can always get an exception out, even if we can't
        # extract any debug info.
        serialized_e = subprocess_rpc.SerializedException.from_exception(e=None, tb=None)
        with self.assertRaises(subprocess_rpc.UnserializableException):
            subprocess_rpc.SerializedException.raise_from(serialized_e)

        class MyException(Exception):
            pass

        class MyIOError(IOError):
            pass

        self._test_raise(MyException, subprocess_rpc.UnserializableException)
        self._test_raise(MyIOError, subprocess_rpc.UnserializableException)

    def test_serializable(self) -> None:
        self._test_raise_builtin(Exception)
        self._test_raise_builtin(AssertionError)
        self._test_raise_builtin(IOError)
        self._test_raise_builtin(NameError)
        self._test_raise_builtin(ValueError)


if __name__ == '__main__':
    run_tests()
