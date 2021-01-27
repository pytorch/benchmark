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
    success, errmsg = pip_install_requirements()
    if not success:
        print("Failed to install torchbenchmark requirements:")
        print(errmsg)
        sys.exit(-1)
    setup(verbose=True)
