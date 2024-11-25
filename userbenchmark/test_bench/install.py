import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "models",
    nargs="*",
    default=[],
    help="Specify one or more models to install. If not set, install all models.",
)
parser.add_argument("--skip", nargs="*", default=[], help="Skip models to install.")
parser.add_argument("--canary", action="store_true", help="Install canary model.")
args, extra_args = parser.parse_known_args()


def install_test_bench_requirements():
    from torchbenchmark import _filter_model_paths

    model_paths = _filter_model_paths(args.models, args.skip, args.canary)
    for path in model_paths:
        print(f"Installing {os.path.basename(path)}...")
        install_py_path = os.path.join(path, "install.py")
        requirements_txt_path = os.path.join(path, "requirements.txt")
        if os.path.exists(install_py_path):
            subprocess.check_call([sys.executable, install_py_path], cwd=path)
        elif os.path.exists(requirements_txt_path):
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "-r",
                    f"{requirements_txt_path}",
                ],
                cwd=path,
            )
        else:
            print(f"SKipped: {os.path.basename(path)}")


if __name__ == "__main__":
    install_test_bench_requirements()
