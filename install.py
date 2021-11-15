import argparse
import subprocess
import os
import sys
import tarfile
from torchbenchmark import setup, _test_https, proxy_suggestion

def git_lfs_checkout():
    tb_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        subprocess.check_call(['git', 'lfs', 'install'], stdout=subprocess.PIPE,
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

def pip_install_requirements():
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return (False, e.output)
    except Exception as e:
        return (False, e)
    return True, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue_on_fail", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("checking out Git LFS files...", end="", flush=True)
    success, errmsg = git_lfs_checkout()
    if success:
        print("OK")
    else:
        print("FAIL")
        print("Failed to checkout git lfs files. Please make sure you have installed git lfs.")
        print(errmsg)
        exit(-1)
    decompress_input()

    success, errmsg = pip_install_requirements()
    if not success:
        print("Failed to install torchbenchmark requirements:")
        print(errmsg)
        if not args.continue_on_fail:
            sys.exit(-1)
    success &= setup(verbose=args.verbose, continue_on_fail=args.continue_on_fail)
    if not success:
        if args.continue_on_fail:
            print("Warning: some benchmarks were not installed due to failure")
        else:
            raise RuntimeError("Failed to complete setup")
