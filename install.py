import argparse
import subprocess
import os
import sys
import yaml
import tarfile
from utils import TORCH_DEPS, proxy_suggestion, get_pkg_versions, _test_https
from pathlib import Path
REPO_ROOT = Path(__file__).parent

def s3_checkout():
    S3_URL_BASE = "https://ossci-datasets.s3.amazonaws.com/torchbench"
    data_dir = REPO_ROOT.joinpath("torchbenchmark", "data")
    model_dir = REPO_ROOT.joinpath("torchbenchmark", "models")
    index_file = REPO_ROOT.joinpath("torchbenchmark", "data", "index.yaml")
    import requests
    with open(index_file, "r") as ind:
        index = yaml.safe_load(ind)
    for input_file in index["INPUT_TARBALLS"]:
        s3_url = f"{S3_URL_BASE}/data/{input_file}"
        r = requests.get(s3_url, allow_redirects=True)
        with open(str(data_dir.joinpath(input_file)), "wb") as output:
            print(f"Checking out {s3_url} to {data_dir.joinpath(input_file)}")
            output.write(r.content)
    for model_file in index["MODEL_PKLS"]:
        s3_url = f"{S3_URL_BASE}/models/{model_file}"
        r = requests.get(s3_url, allow_redirects=True)
        with open(str(model_dir.joinpath(model_file)), "wb") as output:
            print(f"Checking out {s3_url} to {model_dir.joinpath(model_file)}")
            output.write(r.content)

def git_lfs_checkout():
    tb_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        # forcefully install git-lfs to the repo
        subprocess.check_call(['git', 'lfs', 'install', '--force'], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, cwd=tb_dir)
        subprocess.check_call(['git', 'lfs', 'fetch'], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, cwd=tb_dir)
        subprocess.check_call(['git', 'lfs', 'checkout', '.'], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, cwd=tb_dir)
    except subprocess.CalledProcessError as e:
        return (False, e.output)
    except Exception as e:
        return (False, e)
    return True, None

def decompress_input():
    tb_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(tb_dir, "torchbenchmark", "data")
    # Hide decompressed file in .data directory so that they won't be checked in
    decompress_dir = os.path.join(data_dir, ".data")
    os.makedirs(decompress_dir, exist_ok=True)
    # Decompress every tar.gz file
    for tarball in filter(lambda x: x.endswith(".tar.gz"), os.listdir(data_dir)):
        tarball_path = os.path.join(data_dir, tarball)
        print(f"decompressing input tarball: {tarball}...", end="", flush=True)
        tar = tarfile.open(tarball_path)
        tar.extractall(path=decompress_dir)
        tar.close()
        print("OK")

def pip_install_requirements(requirements_txt="requirements.txt"):
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements_txt],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return (False, e.output)
    except Exception as e:
        return (False, e)
    return True, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='*', default=[],
                        help="Specify one or more models to install. If not set, install all models.")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode and check package versions")
    parser.add_argument("--continue_on_fail", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--component", choices=["distributed"], help="Install requirements for optional components.")
    args = parser.parse_args()

    os.chdir(os.path.realpath(os.path.dirname(__file__)))

    print(f"checking packages {', '.join(TORCH_DEPS)} are installed...", end="", flush=True)
    try:
        versions = get_pkg_versions(TORCH_DEPS)
    except ModuleNotFoundError as e:
        print("FAIL")
        print(f"Error: Users must first manually install packages {TORCH_DEPS} before installing the benchmark.")
        sys.exit(-1)
    print("OK")

    print("checking out Git LFS files...", end="", flush=True)
    success, errmsg = git_lfs_checkout()
    if success:
        print("OK")
    else:
        print("FAIL")
        print("git lfs failed. checking out input files from S3...", end="", flush=True)
        s3_checkout()
        print("OK")

    decompress_input()

    if args.component == "distributed":
        success, errmsg = pip_install_requirements(requirements_txt="torchbenchmark/util/distributed/requirements.txt")
        if not success:
            print("Failed to install torchbenchmark distributed requirements:")
            print(errmsg)
            if not args.continue_on_fail:
                sys.exit(-1)
        sys.exit(0)

    success, errmsg = pip_install_requirements()
    if not success:
        print("Failed to install torchbenchmark requirements:")
        print(errmsg)
        if not args.continue_on_fail:
            sys.exit(-1)
    new_versions = get_pkg_versions(TORCH_DEPS)
    if versions != new_versions:
        print(f"The torch packages are re-installed after installing the benchmark deps. \
                Before: {versions}, after: {new_versions}")
        sys.exit(-1)
    from torchbenchmark import setup
    success &= setup(models=args.models, verbose=args.verbose, continue_on_fail=args.continue_on_fail, test_mode=args.test_mode)
    if not success:
        if args.continue_on_fail:
            print("Warning: some benchmarks were not installed due to failure")
        else:
            raise RuntimeError("Failed to complete setup")
