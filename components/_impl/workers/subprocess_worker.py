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


def anonymize_snippet(snippet: str) -> str:
    return f"""
try:
    def _subprocess_anonymous_snippet_f():
{textwrap.indent(textwrap.dedent(snippet).strip(), " " * 8)}
    _subprocess_anonymous_snippet_f()
finally:
    try:
        del _subprocess_anonymous_snippet_f
    except NameError:
        pass  # function definition failed, nothing to cleanup
""".strip()


class SubprocessWorker(base.WorkerBase):
    """Open a subprocess using `python -i`, and use it to execute code.

    The launch command is determined by the `args` property so that subclasses
    can override (generally suppliment) the process launch command. This class
    handles the complexity of communication and fault handling.
    """

    _working_dir: str

    def __init__(self) -> None:
        super().__init__()

        self._stdin: str = os.path.join(self.working_dir, "stdin.log")
        pathlib.Path(self._stdin).touch()

        self._stdout_f: io.FileIO = io.FileIO(
            os.path.join(self.working_dir, "stdout.txt"), mode="w",
        )
        self._stderr_f: io.FileIO = io.FileIO(
            os.path.join(self.working_dir, "stderr.txt"), mode="w",
        )

        self.r_input, self.w_input = os.pipe()
        self.r_output, self.w_output = os.pipe()
        self.r_load, self.w_load = os.pipe()

        child_fds = [self.r_input, self.w_output, self.w_load]
        if subprocess_rpc.IS_WINDOWS:
            import msvcrt
            self._fd_to_handle = {
                fd: msvcrt.get_osfhandle(fd)
                for fd in child_fds
            }

            for fd in child_fds:
                os.set_inheritable(fd, True)

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.lpAttributeList["handle_list"].extend(
                self._fd_to_handle.values())

            popen_kwargs = {
                "startupinfo": startupinfo,
            }

        else:
            self._fd_to_handle = {fd: fd for fd in child_fds}
            popen_kwargs = {
                "close_fds": True,
                "pass_fds": child_fds,
            }

        self._proc = subprocess.Popen(
            args=self.args,
            stdin=subprocess.PIPE,
            stdout=self._stdout_f,
            stderr=self._stderr_f,
            encoding=subprocess_rpc.ENCODING,
            bufsize=1,
            cwd=os.getcwd(),
            **popen_kwargs,
        )

        self._worker_bootstrap_finished: bool = False
        self._bootstrap_worker()

    def _bootstrap_worker(self) -> None:
        """Import subprocess_rpc in `self._proc`.

        `_run` relies on `subprocess_rpc` for communication and
        error handling, so if the import fails it will deadlock. Instead we
        need to do an initial import with a timeout so that we can surface
        failures to users.
        """
        self.write_stdin(anonymize_snippet(f"""
            try:
                import os
                import sys
                if not sys.path[0]:
                    sys.path[0] = {repr(sys.path[0])}
                cwd = os.getcwd()
                sys.path.append(cwd)
                from components._impl.workers import subprocess_rpc
                assert sys.path.pop() == cwd
                globals()["subprocess_rpc"] = subprocess_rpc
                subprocess_rpc._write_to_pipe(
                    {self._fd_to_handle[self.w_output]}, b"IMPORT_SUCCESS")
            except:
                sys.exit(1)
        """))

        with self.watch_stdout_stderr() as get_output:
            try:
                result = subprocess_rpc._read_from_pipe(r_fd=self.r_output)
                assert result == b"IMPORT_SUCCESS"
                self._worker_bootstrap_finished = True
            except:
                stdout, stderr = get_output()
                cause = "import failed" if self._proc.poll() else "timeout"
                raise RuntimeError(
                    f"Failed to bootstrap worker ({cause}):\n"
                    f"    stdout:\n{textwrap.indent(stdout, ' ' * 8)}\n\n"
                    f"    stderr:\n{textwrap.indent(stderr, ' ' * 8)}"
                )


    def write_stdin(self, msg: str) -> None:
        if self._proc.poll() is not None:
            raise ValueError("`self._proc` has exited. Cannot write to stdin.")

        # Log stdin for debugging. (With time added for convenience.)
        with open(self._stdin, "at", encoding="utf-8") as f:
            now = datetime.datetime.now().strftime("[%Y-%m-%d] %H:%M:%S.%f")
            f.write(f"# {now}\n{msg}\n")

        # Actually write to proc stdin. Python is funny about input; if there
        # aren't enough newlines (contextual based on AST) it will wait rather
        # than executing. To guard against this we liberally apply newlines to
        # avoid ambiguity.
        self._write_stdin_raw(f"\n\n{msg}\n\n")

    def _write_stdin_raw(self, msg: str) -> None:
        proc_stdin = self._proc.stdin
        assert proc_stdin is not None
        proc_stdin.write(msg)
        proc_stdin.flush()

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

    @property
    def in_process(self) -> bool:
        return False

    def run(self, snippet: str) -> None:
        self._run(textwrap.dedent(snippet))

    def store(self, name: str, value: typing.Any, in_memory: bool = False) -> None:
        if in_memory:
            raise NotImplementedError("SubprocessWorker does not support `in_memory`")

        # NB: we convert the bytes to a hex string to avoid encoding issues.
        self._run(anonymize_snippet(f"""
            import marshal
            globals()[{repr(name)}] = marshal.loads(bytes.fromhex(
                {repr(marshal.dumps(value).hex())}
            ))
        """))

    def load(self, name: str) -> typing.Any:
        # It is important to scope the file write through
        # `_subprocess_impl_load_fn`, because otherwise we leak the file
        # descriptor.
        self._run(anonymize_snippet(f"""
            import marshal
            subprocess_rpc._write_to_pipe(
                {self._fd_to_handle[self.w_load]},
                marshal.dumps({name}),
            )
        """))

        return marshal.loads(subprocess_rpc._read_from_pipe(r_fd=self.r_load))

    def _run(self, snippet: str) -> None:
        """Helper method for running code in a subprocess."""
        assert self._worker_bootstrap_finished

        with self.watch_stdout_stderr() as get_output:
            subprocess_rpc._write_to_pipe(
                self.w_input, snippet.encode(subprocess_rpc.ENCODING))

            self.write_stdin(textwrap.dedent(f"""
                subprocess_rpc.run_block(
                    input_fd={self._fd_to_handle[self.r_input]},
                    output_fd={self._fd_to_handle[self.w_output]},
                )
            """).strip())

            result = marshal.loads(subprocess_rpc._read_from_pipe(r_fd=self.r_output))
            if isinstance(result, str):
                assert result == subprocess_rpc.SUCCESS
                return

            assert isinstance(result, dict)
            serialized_e = subprocess_rpc.SerializedException(**result)
            stdout, stderr = get_output()
            subprocess_rpc.SerializedException.raise_from(
                serialized_e=serialized_e,
                extra_context=(
                    f"    stdout:\n{textwrap.indent(stdout, ' ' * 8)}\n\n"
                    f"    stderr:\n{textwrap.indent(stderr, ' ' * 8)}"
                )
            )

    def __del__(self) -> None:
        if self._proc.poll() is None:
            if not subprocess_rpc.IS_WINDOWS:
                self._proc.send_signal(signal.SIGINT)

            try:
                self._proc.terminate()

            except PermissionError:
                # NoisePoliceWorker runs under sudo, and thus will not allow
                # SIGTERM to be sent. Note that because __del__ is sometimes
                # called during program shutdown, we can't use
                # `self.write_stdin`. (Because some of the modules will have
                # already been unloaded.)
                self._write_stdin_raw("exit()\n")

        # Unfortunately Popen does not clean up stdin when using PIPE. However
        # we also can't unconditionally close the fd as it could interfere with
        # the orderly teardown of the process. We try our best to kill
        # `self._proc` in the previous block; here we wait for a reasonable
        # timeout (1 second, which should be more than enough), and then if
        # `self._proc` is terminated we make sure its stdin TextIOWrapper is
        # closed as well.
        self._proc.wait(timeout=1)
        if self._proc.poll() is not None:
            proc_stdin = self._proc.stdin
            if proc_stdin is not None:
                proc_stdin.close()

        # We own these fd's, and it seems that we can unconditionally close
        # them without impacting the shutdown of `self._proc`.
        self._stdout_f.close()
        self._stderr_f.close()

        # Close our communication fd's.
        os.close(self.r_input)
        os.close(self.w_input)
        os.close(self.r_output)
        os.close(self.w_output)
        os.close(self.r_load)
        os.close(self.w_load)

        # Finally, make sure we don't leak any files.
        shutil.rmtree(self._working_dir, ignore_errors=True)
