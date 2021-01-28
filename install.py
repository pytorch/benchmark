import argparse
import subprocess
import sys
from torchbenchmark import setup, _test_https, proxy_suggestion


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

    success, errmsg = pip_install_requirements()
    if not success:
        print("Failed to install torchbenchmark requirements:")
        print(errmsg)
        if not args.continue_on_fail:
            sys.exit(-1)
    success &= setup(verbose=args.verbose, continue_on_fail=args.continue_on_fail)
    if not success:
        raise RuntimeError("Failed to complete setup")
