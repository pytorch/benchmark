import subprocess
import sys
import os
from subprocess import PIPE

class BenchRunner(object):
    def __init__(self, conda_env_path, conda_root):

        new_env = {
            "CONDA_SHLVL": "1",
            "CONDA_EXE": os.path.join(conda_root, "bin/conda"),
            "CONDA_PYTHON_EXE": os.path.join(conda_root, "bin/python"),
            "CONDA_PROMPT_MODIFIER": f"({conda_env_path})",
            "_CE_CONDA": "",
            "CONDA_PREFIX": conda_env_path,
            "CONDA_DEFAULT_ENV": conda_env_path,
        }
        keys_to_copy = [
            "TMPDIR",
            "PWD",
            "USER",
            "HOME",
            "PATH",
        ]
        for key in keys_to_copy:
            new_env[key] = os.environ[key]

        # this seemed to work at least..
        #subprocess.check_output(["python", "-c", "import sys; print(sys.executable)"],  env=new_env)
        self.proc = subprocess.Popen(['python', 'scripts/benchrunner.py'], 
                                     env=new_env,
                                     stdin=PIPE, stdout=PIPE, stderr=subprocess.STDOUT)
        init_msg = self._recv()
        print(init_msg)

    def _send(self, msg):
        self.proc.stdin.write(f'{msg}\n'.encode('ascii'))
        self.proc.stdin.flush()

    def _recv(self):
        return self.proc.stdout.readline().decode().strip()

    def run_benchmark(self, benchmark):
        self._send(f"run {benchmark}")
        rc = self._recv()
        try:
            runtime = float(rc)
            return runtime
        except:
            raise RuntimeError(f"Failed to run {benchmark}, got {rc}")

    def terminate(self):
        self._send("exit")
        rc = self._recv()
        assert rc == "clean exit"
        return True


if __name__ == "__main__":
    env = "/Users/whc/env_1.7"
    conda_root = "/Users/whc/miniconda3/"
    runner = BenchRunner(env, conda_root)
    print(runner.run_benchmark("blah"))
    print(runner.run_benchmark("another blah"))
    runner.terminate()
    # proc = subprocess.run(["/usr/bin/env", "bash", "-c", "source activate root",
                            #  f"conda activate {env}",
                            #  "python", "scripts/benchrunner.py",
                            #  "conda deactivate"])
   # import ipdb; ipdb.set_trace()
    # print(proc.stdout.readline())
    # proc.stdin.write(b'run blah\n')
    # proc.stdin.flush()
    # print(proc.stdout.readline())
    # proc.stdin.write(b'run another blah\n')
    # proc.stdin.flush()
    # print(proc.stdout.readline())
    # proc.stdin.write(b'exit\n')
    # proc.stdin.flush()
    # print(proc.stdout.readline())
    # # # proc.stdout.flush()
    # print(proc.stdout.readline())
    # import pdb; pdb.set_trace()