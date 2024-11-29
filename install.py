import argparse
import os
import subprocess
import sys
from pathlib import Path

from userbenchmark import list_userbenchmarks
from utils import generate_pkg_constraints, get_pkg_versions, TORCH_DEPS
from utils.python_utils import pip_install_requirements

REPO_ROOT = Path(__file__).parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
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
    parser.add_argument("--skip", nargs="*", default=[], help="Skip models to install.")
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Only require torch to be installed, ignore torchvision and torchaudio.",
    )
    parser.add_argument(
        "--numpy",
        action="store_true",
        help="Only require numpy to be installed, ignore torch, torchvision and torchaudio.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run the version check and generate the contraints",
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

    if args.torch or args.userbenchmark:
        TORCH_DEPS = ["numpy", "torch"]
    if args.numpy:
        TORCH_DEPS = ["numpy"]
    print(
        f"checking packages {', '.join(TORCH_DEPS)} are installed, generating constaints...",
        end="",
        flush=True,
    )
    if args.userbenchmark:
        TORCH_DEPS = ["numpy", "torch"]
    try:
        versions = get_pkg_versions(TORCH_DEPS)
    except ModuleNotFoundError as e:
        print("FAIL")
        print(
            f"Error: Users must first manually install packages {TORCH_DEPS} before installing the benchmark."
        )
        sys.exit(-1)
    generate_pkg_constraints(versions)
    print("OK")

    if args.check_only:
        exit(0)

    if args.userbenchmark:
        # Install userbenchmark dependencies if exists
        userbenchmark_dir = REPO_ROOT.joinpath("userbenchmark", args.userbenchmark)
        cmd = [sys.executable, "install.py"]
        print(
            f"Installing userbenchmark {args.userbenchmark} with extra args: {extra_args}"
        )
        if args.models:
            cmd.extend(["--models"] + args.models)
        if args.skip:
            cmd.extend(["--skip"] + args.skip)
        if args.canary:
            cmd.extend(["--canary"])
        if args.continue_on_fail:
            cmd.extend(["--continue_on_fail"])
        cmd.extend(extra_args)
        if userbenchmark_dir.joinpath("install.py").is_file():
            # add the current run env to PYTHONPATH to load framework install utils
            run_env = os.environ.copy()
            run_env["PYTHONPATH"] = Path(REPO_ROOT).as_posix()
            subprocess.check_call(
                cmd,
                cwd=userbenchmark_dir.absolute(),
                env=run_env,
            )
        sys.exit(0)

    success, errmsg = pip_install_requirements(continue_on_fail=True)
    if not success:
        print("Failed to install torchbenchmark requirements:")
        print(errmsg)
        if not args.continue_on_fail:
            sys.exit(-1)
    from torchbenchmark import setup

    success &= setup(
        models=args.models,
        skip_models=args.skip,
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
            f"The numpy and torch package versions become inconsistent after installing the benchmark deps. \
                Before: {versions}, after: {new_versions}"
        )
        sys.exit(-1)
    else:
        print(f"installed torchbench with package constraints: {versions}")
