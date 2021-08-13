"""Unit tests specifically for the components of SubprocessWorker.

End-to-end tests (e.g. does SubprocessWorker properly implement the
WorkerBase API) still live in `test_worker`.
"""

import functools
import os
import textwrap
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


INDENT = " " * 12


class TestParseFunction(TestCase):

    def test_parse_trivial(self) -> None:
        def f(x: int) -> None:
            pass

        _, body = task_base.parse_f(f)
        self.assertExpectedInline(
            textwrap.indent(body, INDENT), """\
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
            textwrap.indent(body, INDENT), """\
            for _ in range(10):
                pass""",
        )

    def test_parse_inline(self) -> None:
        def f(x: typing.Any, y: int = 1) -> None: print([x for _ in range(y)])

        _, body = task_base.parse_f(f)
        self.assertExpectedInline(
            textwrap.indent(body, INDENT), """\
            print([x for _ in range(y)])""",
        )

    def test_parse_with_comments(self) -> None:
        def f(
            x: int,   # This is a comment
            y: bool,  # also a comment

            # More comments.
        ) -> typing.Any:  # Comment on return line.
            """Docstring

            Note: This will be dropped. See `parse_f` for details.
            """

            x += 1

            y = """
                This is preserved.
            """

            # Comment in src.
            return y

        _, body = task_base.parse_f(f)
        self.assertExpectedInline(
            textwrap.indent(body, INDENT), """\
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
            textwrap.indent(body, INDENT), """\
            \"\"\"Identity, but with more steps\"\"\"
            return x""",
        )

        _, body = task_base.parse_f(MyClass.g)
        self.assertExpectedInline(
            textwrap.indent(body, INDENT), """\
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
            # Trailing comment isn't part of the execution, so it is actually
            # dropped by `inspect`. (Surprisingly.)

        _, body = task_base.parse_f(f)
        self.assertExpectedInline(
            textwrap.indent(body, INDENT), """\
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


if __name__ == '__main__':
    run_tests()
