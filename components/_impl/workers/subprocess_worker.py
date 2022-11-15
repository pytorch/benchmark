import contextlib
import datetime
import io
import os
import marshal
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import typing

import components
from components._impl.workers import base
from components._impl.workers import subprocess_rpc


class SubprocessWorker(base.WorkerBase):
    """Open a subprocess using `python -i`, and use it to execute code.

        This class wraps a subprocess which runs a clean instance of Python.
    This enables hermetic execution of stateful code, GIL free concurrent
    benchmarking, and easy use of command line tools from Python.

        When using SubprocessWorker, it is important to remember that while the
    environment is (or at least tries to be) identical to the parent, it does
    not share state or initialization with the parent process. Imports must be
    re-run in the worker, and shared resources (such as file descriptors) will
    not be available. For most applications this mirrors the semantics of
    `timeit.Timer`.

        The principle extension point for SubprocessWorker is the `args`
    property. By overriding it, subclasses can change the nature of the
    underlying subprocess while reusing all of the generic communication and
    fault handling facilities of the base class. For example, suppose we want
    to use TaskSet to pin the worker to a single core. The code is simply:

    ```
        class TasksetZeroWorker(SubprocessWorker):

            @property
            def args(self) -> typing.List[str]:
                return ["taskset", "--cpu-list", "0"] + super().args
    ```
    """

    _working_dir: str
    _alive: bool = False
    _bootstrap_timeout: int = 10  # seconds

    def __init__(self, timeout: typing.Optional[float] = None, extra_env: typing.Optional[typing.Dict[str, str]]=None) -> None:
        super().__init__()

        # Log inputs and outputs for debugging.
        self._command_log = os.path.join(self.working_dir, "commands.log")
        pathlib.Path(self._command_log).touch()

        self._stdout_f: io.FileIO = io.FileIO(
            os.path.join(self.working_dir, "stdout.txt"), mode="w",
        )
        self._stderr_f: io.FileIO = io.FileIO(
            os.path.join(self.working_dir, "stderr.txt"), mode="w",
        )

        # `self._run` has strong assumptions about how `_input_pipe` and
        # `_output_pipe` are used. They should not be accessed in any other
        # context. (The same is true for `self.load` and `_load_pipe`.)
        self._input_pipe = subprocess_rpc.Pipe()
        self._output_pipe = subprocess_rpc.Pipe(
            timeout=timeout,
            timeout_callback=self._kill_proc,
        )
        self._load_pipe = subprocess_rpc.Pipe(
            timeout=timeout,
            timeout_callback=self._kill_proc,
        )

        # Windows and Unix differ in how pipes are shared with children.
        # In Unix they are inherited, while in Windows the child consults the
        # OS to get access. Most of this complexity is handled by
        # `subprocess_rpc.Pipe`, however we also have to make sure Popen
        # exposes the pipes in a platform appropriate way.
        child_fds = [
            self._input_pipe.read_fd,
            self._output_pipe.write_fd,
            self._load_pipe.write_fd,
        ]
        if subprocess_rpc.IS_WINDOWS:
            for fd in child_fds:
                os.set_inheritable(fd, True)

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.lpAttributeList["handle_list"].extend(
                [subprocess_rpc.to_handle(fd) for fd in child_fds])

            popen_kwargs = {
                "startupinfo": startupinfo,
            }

        else:
            popen_kwargs = {
                "close_fds": True,
                "pass_fds": child_fds,
            }

        worker_env = os.environ.copy()
        if extra_env:
            worker_env.update(extra_env)
        self._proc = subprocess.Popen(
            args=self.args,
            stdin=subprocess.PIPE,
            stdout=self._stdout_f,
            stderr=self._stderr_f,
            env=worker_env,
            encoding=subprocess_rpc.ENCODING,
            bufsize=1,
            cwd=os.getcwd(),
            **popen_kwargs,
        )

        # setup the pid of child process in the output pipe
        self._output_pipe.set_writer_pid(self._proc.pid)

        self._worker_bootstrap_finished: bool = False
        self._bootstrap_worker()
        self._alive = True

    @property
    def working_dir(self) -> str:
        # A subclass might need to access `self.working_dir` before calling
        # `super().__init__` in order to properly construct `args`, so we need
        # to lazily initialize it.
        if getattr(self, "_working_dir", None) is None:
            self._working_dir = tempfile.mkdtemp()
        return self._working_dir

    @property
    def args(self) -> typing.List[str]:
        return [sys.executable, "-i", "-u"]

    def run(self, snippet: str) -> None:
        self._run(snippet)

    def store(self, name: str, value: typing.Any, in_memory: bool = False) -> None:
        if in_memory:
            raise NotImplementedError("SubprocessWorker does not support `in_memory`")

        # NB: we convert the bytes to a hex string to avoid encoding issues.
        self._run(f"""
            {name} = {subprocess_rpc.WORKER_IMPL_NAMESPACE}["marshal"].loads(
                bytes.fromhex({repr(marshal.dumps(value).hex())})
            )
        """)

    def load(self, name: str) -> typing.Any:
        self._run(f"""
            {subprocess_rpc.WORKER_IMPL_NAMESPACE}["load_pipe"].write(
                {subprocess_rpc.WORKER_IMPL_NAMESPACE}["marshal"].dumps({name})
            )
        """)

        return marshal.loads(self._load_pipe.read())

    @property
    def in_process(self) -> bool:
        return False

    @property
    def alive(self) -> bool:
        return self._alive and self._proc.poll() is None

    def _bootstrap_worker(self) -> None:
        """Import subprocess_rpc in the worker, and start the work loop.

        Commands are executed by writing to `self._input_pipe`, and waiting for
        a response on `self._output_pipe`. This presumes, however, that there
        is a worker doing the opposite: listening to the input pipe and writing
        to the output pipe. At startup `self._proc` is a simple interactive
        Python process, so we have to bootstrap it to start the work loop or
        else `self._run` will hang waiting for jobs to be processed.
        """

        # NB: This gets sent directly to `self._proc`'s stdin, so it MUST be
        #     a single expression and may NOT contain any empty lines. (Due to
        #     how Python processes commands.)
        bootstrap_command = textwrap.dedent(f"""
            try:
                import marshal
                import sys
                sys_path_old = list(sys.path)
                sys.path = marshal.loads(
                    bytes.fromhex({repr(marshal.dumps(sys.path).hex())})
                )
                # The parent gets priority, but a subclass could set PYTHONPATH
                # so we have to respect extra paths.
                sys.path.extend([i for i in sys_path_old if i and i not in sys.path])
                from components._impl.workers import subprocess_rpc
                output_pipe = subprocess_rpc.Pipe(
                    write_handle={self._output_pipe.write_handle})
                output_pipe.write(subprocess_rpc.BOOTSTRAP_IMPORT_SUCCESS)
                subprocess_rpc.run_loop(
                    input_handle={self._input_pipe.read_handle},
                    output_pipe=output_pipe,
                    load_handle={self._load_pipe.write_handle},
                )
            except:
                sys.exit(1)
        """).strip()

        if self._proc.poll() is not None:
            raise ValueError("Process has already exited.")

        proc_stdin = self._proc.stdin
        assert proc_stdin is not None

        self._log_cmd(bootstrap_command)

        # We need two newlines for Python to stop waiting for more input.
        proc_stdin.write(f"{bootstrap_command}\n\n")
        proc_stdin.flush()

        with self.watch_stdout_stderr() as get_output:
            try:
                # Bootstrapping is very fast. (Unlike user code where we have
                # no a priori expected upper bound.) If we don't get a response
                # prior to the timeout, it is overwhelmingly likely that the
                # worker died or the bootstrap failed. (E.g. failed to resolve
                # import path.) This simply allows us to raise a good error.
                bootstrap_pipe = subprocess_rpc.Pipe(
                    read_handle=self._output_pipe.read_handle,
                    write_handle=self._output_pipe.write_handle,
                    timeout=self._bootstrap_timeout,
                )
                result = bootstrap_pipe.read()
                assert result == subprocess_rpc.BOOTSTRAP_IMPORT_SUCCESS, result

                result = bootstrap_pipe.read()
                assert result == subprocess_rpc.BOOTSTRAP_INPUT_LOOP_SUCCESS, result

                self._worker_bootstrap_finished = True
                assert self._proc.poll() is None
            except (Exception, KeyboardInterrupt) as e:
                stdout, stderr = get_output()
                cause = "import failed" if self._proc.poll() else "timeout"
                raise e from RuntimeError(
                    f"Failed to bootstrap worker ({cause}):\n"
                    f"    working_dir: {self.working_dir}\n"
                    f"    stdout:\n{textwrap.indent(stdout, ' ' * 8)}\n\n"
                    f"    stderr:\n{textwrap.indent(stderr, ' ' * 8)}"
                )

    def _log_cmd(self, snippet: str) -> None:
        with open(self._command_log, "at", encoding="utf-8") as f:
            now = datetime.datetime.now().strftime("[%Y-%m-%d] %H:%M:%S.%f")
            f.write(f"# {now}\n{snippet}\n\n")

    @contextlib.contextmanager
    def watch_stdout_stderr(self):
        # Get initial state for stdout and stderr, since we only want to
        # capture output since the contextmanager started.
        stdout_stat = os.stat(self._stdout_f.name)
        stderr_stat = os.stat(self._stderr_f.name)

        def get() -> typing.Tuple[str, str]:
            with open(self._stdout_f.name, "rb") as f:
                _ = f.seek(stdout_stat.st_size)
                stdout = f.read().decode("utf-8").strip()

            with open(self._stderr_f.name, "rb") as f:
                _ = f.seek(stderr_stat.st_size)
                stderr = f.read().decode("utf-8").strip()

            return stdout, stderr

        yield get

    def _run(self, snippet: str) -> None:
        """Helper method for running code in a subprocess."""
        assert self._worker_bootstrap_finished
        assert self.alive, "Process has exited"

        snippet = textwrap.dedent(snippet)
        with self.watch_stdout_stderr() as get_output:
            self._input_pipe.write(snippet.encode(subprocess_rpc.ENCODING))
            self._log_cmd(snippet)

            result = marshal.loads(self._output_pipe.read())
            if isinstance(result, str):
                assert result == subprocess_rpc.SUCCESS
                return

            assert isinstance(result, dict)
            if not result:
                stdout, stderr = get_output()
                raise subprocess.SubprocessError(
                    "Uncaught Exception in worker:"
                    f"    working_dir: {self.working_dir}\n"
                    f"    stdout:\n{textwrap.indent(stdout, ' ' * 8)}\n\n"
                    f"    stderr:\n{textwrap.indent(stderr, ' ' * 8)}")

            serialized_e = subprocess_rpc.SerializedException(**result)
            stdout, stderr = get_output()
            subprocess_rpc.SerializedException.raise_from(
                serialized_e=serialized_e,
                extra_context=(
                    f"    working_dir: {self.working_dir}\n"
                    f"    stdout:\n{textwrap.indent(stdout, ' ' * 8)}\n\n"
                    f"    stderr:\n{textwrap.indent(stderr, ' ' * 8)}"
                )
            )

    def _kill_proc(self) -> None:
        """Best effort to kill subprocess."""
        if getattr(self, "_proc", None) is None:
            # We failed in the constructor, so there's nothing to clean up.
            return

        self._input_pipe.write(subprocess_rpc.HARD_EXIT)
        try:
            self._proc.wait(timeout=1)

        except subprocess.TimeoutExpired:
            if not subprocess_rpc.IS_WINDOWS:
                self._proc.send_signal(signal.SIGINT)

            try:
                self._proc.terminate()

            except PermissionError:
                # NoisePoliceWorker runs under sudo, and thus will not allow
                # SIGTERM to be sent.
                print(f"Failed to clean up process {self._proc.pid}")

        # Unfortunately Popen does not clean up stdin when using PIPE. However
        # we also can't unconditionally close the fd as it could interfere with
        # the orderly teardown of the process. We try our best to kill
        # `self._proc` in the previous block; if `self._proc` is terminated we
        # make sure its stdin TextIOWrapper is closed as well.
        try:
            self._proc.wait(timeout=1)
            proc_stdin = self._proc.stdin
            if proc_stdin is not None:
                proc_stdin.close()
        except subprocess.TimeoutExpired:
            pass

        self._alive = False

    def __del__(self) -> None:
        self._kill_proc()

        # We own these fd's, and it seems that we can unconditionally close
        # them without impacting the shutdown of `self._proc`.
        self._stdout_f.close()
        self._stderr_f.close()

        # Finally, make sure we don't leak any files.
        shutil.rmtree(self._working_dir, ignore_errors=True)
