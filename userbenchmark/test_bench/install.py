import argparse
import os
import subprocess
import sys

from typing import Optional, Tuple

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "models",
    nargs="*",
    default=[],
    help="Specify one or more models to install. If not set, install all models.",
)
parser.add_argument("--skip", nargs="*", default=[], help="Skip models to install.")
parser.add_argument("--canary", action="store_true", help="Install canary model.")
parser.add_argument("--continue_on_fail", action="store_true")
args, extra_args = parser.parse_known_args()


def _run(*popenargs, cwd: str, continue_on_fail=False) -> Tuple[bool, Optional[str]]:
    if not continue_on_fail:
        subprocess.check_call(popenargs, cwd=cwd)
        return True, None
    try:
        subprocess.run(
            popenargs,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.output
    except Exception as e:
        return False, str(e)


def run_install_py(cwd: str, continue_on_fail=False) -> Tuple[bool, Optional[str]]:
    cmd = [sys.executable, "install.py"]
    return _run(*cmd, cwd=cwd, continue_on_fail=continue_on_fail)


def pip_install_requirements(cwd: str, continue_on_fail=False) -> Tuple[bool, Optional[str]]:
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    return _run(*cmd, cwd=cwd, continue_on_fail=continue_on_fail)


def install_test_bench_requirements():
    from torchbenchmark import _filter_model_paths

    model_paths = _filter_model_paths(args.models, args.skip, args.canary)
    for path in model_paths:
        print(f"Installing {os.path.basename(path)}...", end="", flush=True)
        install_py_path = os.path.join(path, "install.py")
        requirements_txt_path = os.path.join(path, "requirements.txt")

        success = False
        error_msg: str
        continue_on_fail = bool(args.continue_on_fail)
        if os.path.exists(install_py_path):
            success, error_msg = run_install_py(path, continue_on_fail)
        elif os.path.exists(requirements_txt_path):
            success, error_msg = pip_install_requirements(path, continue_on_fail)
        else:
            print("SKipped")
            continue

        if success:
            print("OK")
        else:
            print("Failed")
            print(error_msg)
            if not continue_on_fail:
                sys.exit(-1)


if __name__ == "__main__":
    install_test_bench_requirements()
