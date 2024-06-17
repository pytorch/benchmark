import argparse
import os
import subprocess
import sys
from pathlib import Path

from userbenchmark import list_userbenchmarks
from utils import get_pkg_versions, TORCH_DEPS

REPO_ROOT = Path(__file__).parent


def pip_install_requirements(requirements_txt="requirements.txt"):
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_txt],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        return (False, e.output)
    except Exception as e:
        return (False, e)
    return True, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        default=[],
        help="Specify one or more models to install. If not set, install all models.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode and check package versions",
    )
    parser.add_argument("--canary", action="store_true", help="Install canary model.")
    parser.add_argument("--continue_on_fail", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--userbenchmark",
        choices=list_userbenchmarks(),
        help="Install requirements for optional components.",
    )
    args, extra_args = parser.parse_known_args()

    os.chdir(os.path.realpath(os.path.dirname(__file__)))

    print(
        f"checking packages {', '.join(TORCH_DEPS)} are installed...",
        end="",
        flush=True,
    )
    if args.userbenchmark:
        TORCH_DEPS = ["torch"]
    try:
        versions = get_pkg_versions(TORCH_DEPS)
    except ModuleNotFoundError as e:
        print("FAIL")
        print(
            f"Error: Users must first manually install packages {TORCH_DEPS} before installing the benchmark."
        )
        sys.exit(-1)
    print("OK")

    if args.userbenchmark:
        # Install userbenchmark dependencies if exists
        userbenchmark_dir = REPO_ROOT.joinpath("userbenchmark", args.userbenchmark)
        cmd = [sys.executable, "install.py"]
        cmd.extend(extra_args)
        if userbenchmark_dir.joinpath("install.py").is_file():
            subprocess.check_call(
                cmd, cwd=userbenchmark_dir.absolute()
            )
        sys.exit(0)

    success, errmsg = pip_install_requirements()
    if not success:
        print("Failed to install torchbenchmark requirements:")
        print(errmsg)
        if not args.continue_on_fail:
            sys.exit(-1)
    from torchbenchmark import setup

    success &= setup(
        models=args.models,
        verbose=args.verbose,
        continue_on_fail=args.continue_on_fail,
        test_mode=args.test_mode,
        allow_canary=args.canary,
    )
    if not success:
        if args.continue_on_fail:
            print("Warning: some benchmarks were not installed due to failure")
        else:
            raise RuntimeError("Failed to complete setup")
    new_versions = get_pkg_versions(TORCH_DEPS)
    if versions != new_versions:
        print(
            f"The torch packages are re-installed after installing the benchmark deps. \
                Before: {versions}, after: {new_versions}"
        )
        sys.exit(-1)
