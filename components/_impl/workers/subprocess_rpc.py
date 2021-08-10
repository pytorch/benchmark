"""Utilities to handle communication between parent worker."""
import dataclasses
import datetime
import inspect
import io
import marshal
import os
import pathlib
import pickle
import struct
import sys
import textwrap
import traceback
import types
import typing


# Constants for passing to and from pipes
_CHECK = b"\x00\x00"
_ULL = "Q"  # Unsigned long long
_ULL_SIZE = len(struct.pack(_ULL, 0))
assert _ULL_SIZE == 8

# Text encoding for input commands.
ENCODING = "utf-8"
SUCCESS = "SUCCESS"

IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import msvcrt

# Precompute serialized normal return values
EMPTY_RESULT = marshal.dumps({})
SUCCESS_BYTES = marshal.dumps(SUCCESS)


class ExceptionUnpickler(pickle.Unpickler):

    @classmethod
    def load_bytes(cls, data: bytes) -> typing.Type[Exception]:
        result = cls(io.BytesIO(data)).load()

        # Make sure we have an Exception class, but not an instantiated
        # Exception.
        if not issubclass(result, Exception):
            raise pickle.UnpicklingError(f"{result} is not an Exception")

        if isinstance(result, Exception):
            raise pickle.UnpicklingError(
                f"{result} is an Exception instance, not a class.")

        return result   # type: ignore[no-any-return]

    def find_class(self, module: str, name: str) -> typing.Any:
        if module != "builtins":
            raise pickle.UnpicklingError(f"Invalid object: {module}.{name}")
        return super().find_class(module, name)


class UnserializableException(Exception):

    def __init__(self, type_repr: str, args_repr: str) -> None:
        self.type_repr = type_repr
        self.args_repr = args_repr
        super().__init__(type_repr, args_repr)


class ChildTraceException(Exception):
    pass


@dataclasses.dataclass(init=True, frozen=True)
class SerializedException:
    _is_serializable: bool
    _type_bytes: bytes
    _args_bytes: bytes

    # Fallbacks for UnserializableException
    _type_repr: str
    _args_repr: str

    _traceback_print: str

    @staticmethod
    def from_exception(e: Exception, tb: types.TracebackType) -> "SerializedException":
        """Best effort attempt to serialize Exception.

        Because this will be used to communicate from a subprocess to its
        parent, we want to surface as much information as possible. It is
        not possible to serialize a traceback because it is too intertwined
        with the runtime; however what we really want is the traceback so we
        can print it. We can grab that string and send it without issue.

        ExceptionUnpickler explicitly refuses to load any non-builtin exception
        (for the same reason we prefer `marshal` to `pickle`), so we won't be
        able to serialize all cases. However we don't want to simply give up
        as this will make it difficult for a user to diagnose what's going on.
        So instead we extract what information we can, and raise an
        UnserializableException in the main process with whatever we were able
        to scrape up from the child process.
        """
        try:
            print_file = io.StringIO()
            traceback.print_exception(
                etype=type(e),
                value=e,
                tb=tb,
                file=print_file,
            )
            print_file.seek(0)
            traceback_print: str = print_file.read()

        except Exception:
            traceback_print = textwrap.dedent("""
                Traceback
                    Failed to extract traceback from worker. This is not expected.
            """).strip()

        try:
            args_bytes: bytes = marshal.dumps(e.args)
            type_bytes = pickle.dumps(e.__class__)

            # Make sure we'll be able to get something out on the other side.
            revived_type = ExceptionUnpickler.load_bytes(data=type_bytes)
            revived_e = revived_type(*marshal.loads(args_bytes))
            is_serializable: bool = True

        except Exception:
            is_serializable = False
            args_bytes = b""
            type_bytes = b""

        # __repr__ can contain arbitrary code, so we can't trust it to noexcept.
        def hardened_repr(o: typing.Any) -> str:
            try:
                return repr(o)

            except Exception:
                return "< Unknown >"

        return SerializedException(
            _is_serializable=is_serializable,
            _type_bytes=type_bytes,
            _args_bytes=args_bytes,
            _type_repr=hardened_repr(e.__class__),
            _args_repr=hardened_repr(getattr(e, "args", None)),
            _traceback_print=traceback_print,
        )

    @staticmethod
    def raise_from(
        serialized_e: "SerializedException",
        extra_context: typing.Optional[str] = None,
    ) -> None:
        """Revive `serialized_e`, and raise.

        We raise the revived exception type (if possible) so that any higher
        try catch logic will see the original exception type. In other words:
        ```
            try:
                worker.run("assert False")
            except AssertionError:
                ...
        ```

        will flow identically to:

        ```
            try:
                assert False
            except AssertionError:
                ...
        ```

        If for some reason we can't move the true exception type to the main
        process (e.g. a custom Exception) we raise UnserializableException as
        a fallback.
        """
        if serialized_e._is_serializable:
            revived_type = ExceptionUnpickler.load_bytes(data=serialized_e._type_bytes)
            e = revived_type(*marshal.loads(serialized_e._args_bytes))
        else:
            e = UnserializableException(serialized_e._type_repr, serialized_e._args_repr)

        traceback_str = serialized_e._traceback_print
        if extra_context:
            traceback_str = f"{traceback_str}\n{extra_context}"

        raise e from ChildTraceException(traceback_str)


def _read_from_pipe(r_fd: int, convert_fd: bool = IS_WINDOWS) -> bytes:
    if convert_fd:
        assert IS_WINDOWS
        r_fd = msvcrt.open_osfhandle(r_fd, os.O_RDONLY)

    # Make sure we a starting from a reasonable place.
    check = os.read(r_fd, len(_CHECK))
    if check != _CHECK:
        raise IOError(f"{check} != {_CHECK}")

    msg_size = struct.unpack(_ULL, os.read(r_fd, _ULL_SIZE))[0]
    return os.read(r_fd, msg_size)


def _write_to_pipe(w_fd: int, msg: bytes, convert_fd: bool = IS_WINDOWS) -> None:
    assert isinstance(msg, bytes), msg
    if convert_fd:
        assert IS_WINDOWS
        w_fd = msvcrt.open_osfhandle(w_fd, os.O_WRONLY)
    os.write(w_fd, _CHECK + struct.pack(_ULL, len(msg)) + msg)


def _log_progress(suffix: str) -> None:
    now = datetime.datetime.now().strftime("[%Y-%m-%d] %H:%M:%S.%f")
    print(f"\n{now}: TIMER_SUBPROCESS_{suffix}")


def run_block(
    *,
    input_fd: int,
    output_fd: int,
):
    result = EMPTY_RESULT
    try:
        _log_progress("BEGIN")
        cmd = _read_from_pipe(input_fd).decode(ENCODING)

        # In Python, `global` means global to a module, not global to the
        # program. So if we simply call `globals()`, we will get the globals
        # for this module (which contains lots of implementation details),
        # not the globals from the from the calling context. So instead we grab
        # the calling frame exec with those globals.
        calling_frame = inspect.stack()[1].frame

        exec(  # noqa: P204
            compile(cmd, "<subprocess-worker>", "exec"),
            calling_frame.f_globals,
        )

        _log_progress("SUCCESS")
        result = SUCCESS_BYTES

    except Exception as e:
        tb = sys.exc_info()[2]
        assert tb is not None
        serialized_e = SerializedException.from_exception(e, tb)
        result = marshal.dumps(dataclasses.asdict(serialized_e))
        _log_progress("FAILED")

    finally:
        _write_to_pipe(output_fd, result)
        _log_progress("FINISHED")
        sys.stdout.flush()
        sys.stderr.flush()
