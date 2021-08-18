"""Add Task abstraction to reduce the friction of controlling a remote worker."""
import abc
import ast
import functools
import inspect
import marshal
import textwrap
import typing

from components._impl.workers import base


class TaskBase(abc.ABC):
    """Convenience layer to allow methods to be called in a worker.

    Because workers are stateful, this implicitly assumes that a Task wraps
    a single worker. However `run_in_worker` is largely agnostic; it simply
    calls `self.worker` and dispatches work to whatever Worker is returned.
    """

    @abc.abstractproperty
    def worker(self) -> base.WorkerBase:
        ...


def parse_f(f: typing.Callable) -> typing.Tuple[inspect.Signature, str]:
    """Extract the source code from a callable."""
    if not inspect.isfunction(f):
        raise TypeError(f"Expected function, got {type(f)}. ({f})")

    signature = inspect.signature(f)

    # It isn't strictly necessary for `f` to be completely type annotated,
    # however one of the key advantages of `run_in_worker` over manually
    # passing strings is that it gives the type checker an opportunity to catch
    # errors. (Which may become more difficult once the code has been shipped)
    # to the worker. And because this is a developer rather than user facing
    # API, it isn't a problem to enforce this strict criteria. This also
    # provides some weak protection against decorators. (See below.)
    for arg, arg_parameter in signature.parameters.items():
        if arg_parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                f"Variadic positional argument `*{arg}` not permitted "
                "for `run_in_worker` function.")

        if arg_parameter.kind == inspect.Parameter.VAR_KEYWORD:
            raise TypeError(
                f"Variadic keywork argument `**{arg}` not permitted "
                "for `run_in_worker` function.")

        if arg_parameter.annotation == inspect.Parameter.empty:
            raise TypeError(f"Missing type annotation for parameter `{arg}`")

    if signature.return_annotation == inspect.Parameter.empty:
        raise TypeError("Missing return annotation.")

    #   We serialize the function by grabbing source from the body, so
    # decorators pose a correctness problem. The structure of a decorator is:
    # ```
    #   def my_decorator(wrapped_f: typing.Callable) -> typing.Callable:
    #
    #       @functools.wraps(wrapped_f)  # <- Optional
    #       def inner(*args, **kwargs):  # Most decorators don't know the signature of `wrapped_f`
    #           # Generally a call to `wrapped_f(*args, **kwargs)` appears
    #           # somewhere, though not required.
    #           ...
    #
    #       return g
    # ```
    #   The inclusion or omission of `functools.wraps` is rather important.
    # If included, it will provide the breadcrumbs to map `inner` back to
    # `wrapped_f`, and the `inspect` library (namely `signature` and
    # `getsource`) will parse the "True" function: `wrapped_f`. (This allows,
    # among other things, type checkers to analyze decorated code.) Otherwise
    # `inspect` will stop at `inner`.
    #
    #   In the case that a function is decorated but does not use
    # `functools.wraps`, it is HIGHLY likely that it uses variadic arguments
    # so that it can forward them. (And thus will fail the signature checks)
    # above. If we are passed a function which is decorated with a "proper"
    # decorator, we catch it here by checking the `__wrapped__` property.
    #
    #   The final case of a decorator with a concrete signature but no
    # `functools.wraps` is not detectable (by design) and is thus caveat
    # emptor.
    if getattr(f, "__wrapped__", None):
        raise TypeError(textwrap.dedent("""
            `f` cannot be decorated below `@run_in_worker` (except for
            @staticmethod) because the extraction logic would not carry through
            said decorator(s).

            Ok:
                @my_decorator
                @run_in_worker()
                def foo() -> None:
                    ...

            Not ok:
                @run_in_worker()
                @my_decorator
                def foo() -> None:
                    ...
        """).strip())

    # Dedent, as `f` may have been defined in a scoped context.
    f_src = textwrap.dedent(inspect.getsource(f))

    #   We don't want to be in the business of writing a Python parser.
    # Fortunately our needs are relatively modest: we simply need to run a few
    # sanity checks, and get the position where the body starts. This means
    # that we can rely on the `ast` library to do the heavy lifting, and grab
    # a few key values at the end.
    f_ast = ast.parse(f_src)
    assert len(f_ast.body) == 1
    assert isinstance(f_ast.body[0], ast.FunctionDef)
    assert f_ast.body[0].body

    # For some reason ast one indexes lineno.
    src_lines = f_src.splitlines(keepends=False)

    node: ast.AST
    for node in f_ast.body[0].body:
        # In Python 3.7, there is a bug in `ast` that causes it to incorrectly
        # report the start line of bare multi-line strings:
        #   https://bugs.python.org/issue16806
        # Given that the only use for such strings is a docstring (or multi
        # line comment), we simply elect to skip over them and index on the
        # first node that will give valid indices.
        if node.col_offset == -1:
            assert isinstance(node.value, ast.Str), f"Expected `ast.Str`, got {type(node)}. ({node}) {node.lineno}"
            continue

        raw_body_lines = src_lines[node.lineno - 1:]
        col_offset = node.col_offset
        break

    else:
        raise TypeError("Could not find valid start of body.")

    body_lines: typing.List[str] = []
    for i, l in enumerate(raw_body_lines):
        prefix, suffix = l[:col_offset], l[col_offset:]

        # The first line of the body may overlap with the signature.
        #   e.g. `def f(): pass`
        # For all other lines, the prefix must only be indentation.
        assert not i or not l.strip() or not prefix.strip(), f"{l}, {col_offset}"

        body_lines.append(suffix)
    return signature, "\n".join(body_lines)


def run_in_worker(scoped: bool = True) -> typing.Callable[..., typing.Any]:
    """Decorator to run Task method in worker rather than the caller.

    The Worker + Task model dictates that the caller generates a string of
    Python source code. However this is not particularly ergonomic; there is
    significant tooling (syntax highlighting, type checking, etc.) which is
    lost if code must be provided as a string literal.

    Moreover, moving values from the caller to the worker can be tedious.
    Simply templating them into a string literal is brittle (because __repr__
    may not produce valid source) and may subtly alter the value (e.g. the
    string representation of a float will not produce the same number as the
    original value). `WorkerBase.store` will safely move values, but does not
    alleviate the ergonomic issues.

    Consider the following, where we want the worker to open a file, read up to
    `n` lines, and then return them to the caller. One implementation would be:

    ```
    def head(self, fpath: str, n: int) -> List[str]:
        self.worker.store("fpath", fpath)
        self.worker.store("n", n)
        self.worker.run(textwrap.dedent('''
            lines = []
            with open(fpath, "rt") as f:
                for i, l in enumerate(f):
                    if i == n:
                        break
                    lines.append(l)
        '''))
        return self.worker.load("lines")
    ```

    It works, but it's not very easy to read and leaks lots of variables
    (fpath, n, lines, f, etc.) into the worker's global namespace. This
    decorator allows the following code to be written instead:

    ```
    @run_in_worker(scoped=True)
    def head(fpath: str, n: int) -> List[str]:
        lines = []
        with open(fpath, "rt") as f:
            for i, l in enumerate(f):
                if i == n:
                    break
                lines.append(l)
        return lines
    ```

    Code in the main thread can call `head` just like any normal function, but
    it is executed in the worker. And unlike the first example, we will not
    pollute the global namespace. (This is because `scoped=True`) There are
    three aspects to `run_in_worker`:

        1) Serialize arguments and revive them in the worker.
        2) Extract the function body.
        3) Retrieve the result from the worker.

    All three are entirely mechanical; `run_in_worker` uses Python AST rather
    than raw string parsing, so it is quite robust. Because ambiguity would be
    very difficult to diagnose in this context, `run_in_worker` requires that
    a complete type annotated signature be provided and that there are no
    variadic arguments. (*args or **kwargs) Moreover, it has same restriction
    for inputs and outputs as `store` and `load`: the values must be
    serializable by the `marshal` library. (i.e. basic Python types)
    """

    def outer(f: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
        # This will unwrap the `@staticmethod` descriptor and recover the true f
        #   https://stackoverflow.com/questions/53694087/unwraping-and-wrapping-again-staticmethod-in-meta-class
        #
        # Note: The `@staticmethod` decorator must appear BELOW the
        #       `@run_in_worker` decorator.
        try:
            f = f.__get__(object, None)  # type: ignore[attr-defined]
        except AttributeError:
            pass

        signature, f_body = parse_f(f)
        has_return_value = (signature.return_annotation is not None)
        if has_return_value and not scoped:
            raise TypeError(
                "Unscoped (globally executed) call can not have a return value.")

        @functools.wraps(f)
        def inner(
            self: TaskBase,
            *args: typing.Any,
            **kwargs: typing.Any
        ) -> typing.Any:
            bound_signature = signature.bind(*args, **kwargs)
            bound_signature.apply_defaults()

            body: typing.List[str] = ["# Deserialize args", "import marshal"]
            for arg_name, arg_value in bound_signature.arguments.items():
                try:
                    arg_bytes = marshal.dumps(arg_value)
                except ValueError:
                    raise ValueError(f"unmarshallable arg {arg_name}: {arg_value}")

                body.append(f"{arg_name} = marshal.loads(bytes.fromhex({repr(arg_bytes.hex())}))  # {arg_value}")
            body.extend(["", "# Wrapped source"] + f_body.splitlines(keepends=False))

            src = "\n".join([
                "def _run_in_worker_f():",
                textwrap.indent("\n".join(body), " " * 4),
                textwrap.dedent("""
                try:
                    # Clear prior value if it exists.
                    del _run_in_worker_result

                except NameError:
                    pass

                _run_in_worker_result = _run_in_worker_f()
                """)
            ])

            # `worker.load` is not free, so for void functions we skip it.
            if has_return_value:
                self.worker.run(src)
                return self.worker.load("_run_in_worker_result")

            else:
                src = f"{src}\nassert _run_in_worker_result is None"
                self.worker.run(src)

        return inner
    return outer
