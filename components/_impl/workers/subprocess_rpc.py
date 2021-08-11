"""Utilities to handle communication between parent worker.

This module implements three principle facilities:
    1) Raw IPC (via the Pipe class)
    2) Exception propagation (via the SerializedException class)
    3) A run loop for the worker (via the run_loop function)
"""
import contextlib
import dataclasses
import datetime
import io
import marshal
import os
import pickle
import struct
import sys
import textwrap
import threading
import time
import traceback
import types
import typing


# Shared static values / namespace between worker and parent
BOOTSTRAP_IMPORT_SUCCESS = b"BOOTSTRAP_IMPORT_SUCCESS"
BOOTSTRAP_INPUT_LOOP_SUCCESS = b"BOOTSTRAP_INPUT_LOOP_SUCCESS"
WORKER_IMPL_NAMESPACE = "__worker_impl_namespace"

# Constants for passing to and from pipes
_CHECK = b"\x00\x00"
_TIMEOUT = b"\x01\x01"
assert len(_CHECK) == len(_TIMEOUT)

_ULL = "Q"  # Unsigned long long
_ULL_SIZE = len(struct.pack(_ULL, 0))
assert _ULL_SIZE == 8

# Text encoding for input commands.
ENCODING = "utf-8"
SUCCESS = "SUCCESS"

# In Python, `sys.exit()` is a soft exit. It throws a SystemExit, and only
# exits if that is not caught. `os._exit()` is not catchable, and is suitable
# for cases where we really, really need to exit. This is of particular
# importance because the worker run loop does its very best to swallow
# exceptions.
HARD_EXIT = "import os\nos._exit(0)".encode(ENCODING)

# Precompute serialized normal return values
EMPTY_RESULT = marshal.dumps({})
SUCCESS_BYTES = marshal.dumps(SUCCESS)


# =============================================================================
# == Raw Communication ========================================================
# =============================================================================

# Windows does not allow subprocesses to inherit file descriptors, so instead
# we have to go the the OS and get get the handle for the backing resource.
IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import msvcrt
    def to_handle(fd: typing.Optional[int]) -> typing.Optional[int]:
        return None if fd is None else msvcrt.get_osfhandle(fd)

    def from_handle(handle: typing.Optional[int], mode: int) -> typing.Optional[int]:
        return None if handle is None else msvcrt.open_osfhandle(handle, mode)

else:
    to_handle = lambda fd: fd
    from_handle = lambda fd, _: fd


class _TimeoutPIPE:
    """Allow Pipe to interrupt its read.

    `os.read` is a syscall, which means it is not interruptable. This means
    that normal timeout mechanisms such as `asyncio.wait_for(..., timeout=...)`
    will not work because they rely on the awaited function returning control
    to the event loop. An alternate formulation uses `run_in_executor` and
    `asyncio.wait`, which places the read on a side thread under the hood.
    However this is also not suitable, because:

        1)  This additional machinery increases the cost when data is already
            present in the Pipe (most common case) ~1000x, from O(us) to O(ms)
        2)  We have to poll the future, which wastes the awaitable nature `read`

    Instead of trying to interrupt the pipe read, we can cause it terminate by
    writing to the pipe; because we control the read (via `Pipe.read`) we can
    catch the sentinel timeout value and raise appropriately.

    This class is designed to be extremely lightweight. Timeouts should be on
    the order of seconds (or minutes), and are only to prevent deadlocks in the
    case of catastrophic worker failure. As a result, we prioritize low
    resource usage over the ability to support small timeouts.
    """

    _singleton_lock = threading.Lock()
    _singleton: typing.Optional["_TimeoutPIPE"] = None

    _loop_lock = threading.Lock()
    _active_reads: typing.Dict[int, typing.Tuple[float, float]]
    _loop_cadence = 1  # second

    @classmethod
    def singleton(cls) -> "_TimeoutPIPE":
        # This class will spawn a thread, so we only want one active at a time.
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    def __init__(self) -> None:
        self._active_reads = {}
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def _loop(self):
        # This loop is scoped to the life of the process, so we rely on process
        # teardown to pull the rug out from under the daemonic thread running
        # this function.
        while True:
            time.sleep(self._loop_cadence)
            now = time.time()
            with self._loop_lock:
                for w_fd, (timeout, start_time) in tuple(self._active_reads.items()):
                    if now - start_time >= timeout and w_fd in self._active_reads:
                        os.write(w_fd, _TIMEOUT)
                        self.pop(w_fd)

    def pop(self, w_fd: int) -> None:
        self._active_reads.pop(w_fd, None)

    @classmethod
    @contextlib.contextmanager
    def maybe_timeout_read(cls, pipe: "Pipe") -> None:
        timeout = pipe.timeout

        # Workers should never set a timeout, so in that case we want to exit
        # without calling `cls.singleton()` so we don't spawn an unnecessary
        # loop thread.
        if timeout is None:
            yield
            return

        w_fd = pipe.write_fd
        assert w_fd is not None, "Cannot timeout without write file descriptor."
        singleton = cls.singleton()
        with singleton._loop_lock:
            # This will only occur in the case of concurrent reads on different
            # threads (not supported) or a leaked case.
            assert w_fd not in singleton._active_reads, f"{w_fd} is already being watched."
            singleton._active_reads[w_fd] = (timeout, time.time())

        try:
            yield

        finally:
            singleton.pop(w_fd)


class Pipe:
    """Helper class to move data in a robust fashon.

    This class handles:
        1) File descriptor lifetimes
        2) File descriptor inheritance
        3) Message packing and unpacking
        4) (Optional) timeouts for reads
    """

    def __init__(
        self,
        read_handle: typing.Optional[int] = None,
        write_handle: typing.Optional[int] = None,
        timeout: typing.Optional[float] = None
    ) -> None:
        self._owns_pipe = read_handle is None and write_handle is None
        if self._owns_pipe:
            self.read_fd, self.write_fd = os.pipe()
        else:
            self.read_fd = from_handle(read_handle, os.O_RDONLY)
            self.write_fd = from_handle(write_handle, os.O_WRONLY)

        self.read_handle = read_handle or to_handle(self.read_fd)
        self.write_handle = write_handle or to_handle(self.write_fd)
        self.timeout = timeout

    def _read(self, size: int) -> bytes:
        """Handle the low level details of reading from the PIPE."""
        if self.read_fd is None:
            raise IOError("Cannot read from PIPE, we do not have the read handle")

        with _TimeoutPIPE.maybe_timeout_read(self):
            raw_msg = os.read(self.read_fd, len(_CHECK) + size)

        check_bytes, msg = raw_msg[:len(_CHECK)], raw_msg[len(_CHECK):]
        if check_bytes == _TIMEOUT:
            raise IOError(f"Exceeded timeout: {self.timeout}")

        if check_bytes != _CHECK:
            raise IOError(f"{check} != {_CHECK}, {msg}")

        if len(msg) != size:
            raise IOError(f"len(msg) != size: {len(msg)} vs. {size}")

        return msg

    def read(self) -> bytes:
        msg_size = struct.unpack(_ULL, self._read(_ULL_SIZE))[0]
        return self._read(msg_size)

    def write(self, msg: bytes) -> None:
        if self.write_fd is None:
            raise IOError("Cannot write from PIPE, we do not have the write handle")
        assert isinstance(msg, bytes), msg
        packed_msg = (
            # First read: message length
            _CHECK + struct.pack(_ULL, len(msg)) +

            # Second read: message contents
            _CHECK + msg
        )

        os.write(self.write_fd, packed_msg)

    def __del__(self) -> None:
        if self._owns_pipe:
            os.close(self.read_fd)
            os.close(self.write_fd)


# =============================================================================
# == Exception Propagation  ===================================================
# =============================================================================

class ExceptionUnpickler(pickle.Unpickler):
    """Unpickler which is specialized for Exception types.

    When we catch an exception that we want to raise in another process, we
    need to include the type of Exception. For custom exceptions this is a
    problem, because pickle dynamically resolves imports which means we might
    not be able to unpickle in the parent. (And reviving them by replaying
    the constructor args might not work.) So in the interest of robustness, we
    confine ourselves to builtin Exceptions. (With UnserializableException as
    a fallback.)

    However it is not possible to marshal even builtin Exception types, so
    instead we use pickle and check that the type is builtin in `find_class`.
    """

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
    """Fallback class for if a non-builtin Exception is raised."""

    def __init__(self, type_repr: str, args_repr: str) -> None:
        self.type_repr = type_repr
        self.args_repr = args_repr
        super().__init__(type_repr, args_repr)


class ChildTraceException(Exception):
    """Used to display a raising child's stack trace in the parent's stderr."""

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
        can print it. We can grab that string and send it without issue. (And
        providing a stack trace is very important for debuggability.)

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


# =============================================================================
# == Snippet Execution  =======================================================
# =============================================================================

def _log_progress(suffix: str) -> None:
    now = datetime.datetime.now().strftime("[%Y-%m-%d] %H:%M:%S.%f")
    print(f"{now}: TIMER_SUBPROCESS_{suffix}")


def _run_block(
    *,
    input_pipe: Pipe,
    output_pipe: Pipe,
    globals_dict: typing.Dict[str, typing.Any],
):
    result = EMPTY_RESULT
    try:
        _log_progress("BEGIN_READ")
        cmd = input_pipe.read().decode(ENCODING)
        _log_progress("BEGIN_EXEC")

        exec(  # noqa: P204
            compile(cmd, "<subprocess-worker>", "exec"),
            globals_dict
        )

        _log_progress("SUCCESS")
        result = SUCCESS_BYTES

    except (Exception, KeyboardInterrupt) as e:
        tb = sys.exc_info()[2]
        assert tb is not None
        serialized_e = SerializedException.from_exception(e, tb)
        result = marshal.dumps(dataclasses.asdict(serialized_e))
        _log_progress("FAILED")

    finally:
        output_pipe.write(result)
        _log_progress("FINISHED")
        sys.stdout.flush()
        sys.stderr.flush()


def run_loop(
    *,
    input_handle: int,
    output_pipe: Pipe,
    load_handle: int,
) -> None:
    input_pipe = Pipe(read_handle=input_handle)

    # In general, we want a clean separation between user code and framework
    # code. However, certain methods in SubprocessWorker (store and load)
    # want to access implementation details in this module. As a result, we
    # run tasks through a context where globals start out clean EXCEPT for
    # a namespace where we can stash implementation details.
    globals_dict = {
        WORKER_IMPL_NAMESPACE: {
            "subprocess_rpc": sys.modules[__name__],
            "marshal": marshal,
            "load_pipe": Pipe(write_handle=load_handle)
        }
    }

    output_pipe.write(BOOTSTRAP_INPUT_LOOP_SUCCESS)
    while True:
        _run_block(
            input_pipe=input_pipe,
            output_pipe=output_pipe,
            globals_dict=globals_dict,
        )
