"""
Return a list of recent PyTorch wheels published on download.pytorch.org.
Users can specify package name, python version, platform, and the number of days to return.
If one of the packages specified is missing on one day, the script will skip outputing the results on that day.
"""

import os
import re
import requests
import argparse
import urllib.parse
from datetime import date, timedelta
from bs4 import BeautifulSoup
from collections import defaultdict
import sys
from pathlib import Path
import subprocess

from typing import List

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

with add_path(str(REPO_ROOT)):
    from utils.cuda_utils import DEFAULT_CUDA_VERSION, CUDA_VERSION_MAP
    from utils.python_utils import DEFAULT_PYTHON_VERSION, PYTHON_VERSION_MAP

PYTORCH_CUDA_VERISON = CUDA_VERSION_MAP[DEFAULT_CUDA_VERSION]["pytorch_url"]
PYTORCH_PYTHON_VERSION = PYTHON_VERSION_MAP[DEFAULT_PYTHON_VERSION]["pytorch_url"]

torch_wheel_nightly_base = f"https://download.pytorch.org/whl/nightly/{PYTORCH_CUDA_VERISON}/"
torch_nightly_wheel_index = f"https://download.pytorch.org/whl/nightly/{PYTORCH_CUDA_VERISON}/torch_nightly.html"
torch_nightly_wheel_index_override = "torch_nightly.html" 

def memoize(function):
    """ 
    """
    call_cache = {}

    def memoized_function(*f_args):
        if f_args in call_cache:
            return call_cache[f_args]
        call_cache[f_args] = result = function(*f_args)
        return result

    return memoized_function

@memoize
def get_wheel_index_data(py_version, platform_version, url=torch_nightly_wheel_index, override_file=torch_nightly_wheel_index_override):
    """
    """
    if os.path.isfile(override_file) and os.stat(override_file).st_size:
        with open(override_file) as f:
            data = f.read()
    else:
        r = requests.get(url)
        r.raise_for_status()
        data = r.text
    soup = BeautifulSoup(data, 'html.parser')
    data = defaultdict(dict)
    for link in soup.find_all('a'):
        group_match = re.search("([a-z]*)-(.*)-(.*)-(.*)-(.*)\.whl", link.text)
        # some packages (e.g., torch-rec) doesn't follow this naming convention
        if not group_match:
            continue
        pkg, version, py, py_m, platform = group_match.groups()
        version = urllib.parse.unquote(version)
        if py == py_version and platform == platform_version:
            full_url = os.path.join(torch_wheel_nightly_base, link.text)
            data[pkg][version] = full_url
    return data

def get_nightly_wheel_urls(packages:list, date:date,
                           py_version=PYTORCH_PYTHON_VERSION, platform_version='linux_x86_64'):
    """Gets urls to wheels for specified packages matching the date, py_version, platform_version
    """
    date_str = f"{date.year}{date.month:02}{date.day:02}"
    data = get_wheel_index_data(py_version, platform_version)

    rc = {}
    for pkg in packages:
        pkg_versions = data[pkg]
        # multiple versions could happen when bumping the pytorch version number
        # e.g., both torch-1.11.0.dev20220211%2Bcu113-cp38-cp38-linux_x86_64.whl and
        # torch-1.12.0.dev20220212%2Bcu113-cp38-cp38-linux_x86_64.whl exist in the download link
        keys = sorted([key for key in pkg_versions if date_str in key], reverse=True)
        if len(keys) > 1:
            print(f"Warning: multiple versions matching a single date: {keys}, using {keys[0]}")
        if len(keys) == 0:
            return None
        full_url = pkg_versions[keys[0]]
        rc[pkg] = {
            "version": keys[0],
            "wheel": full_url,
        }
    return rc

def get_nightly_wheels_in_range(packages:list, start_date:date, end_date:date,
                                py_version=PYTORCH_PYTHON_VERSION, platform_version='linux_x86_64', reverse=False):
    rc = []
    curr_date = start_date
    while curr_date <= end_date:
        curr_wheels = get_nightly_wheel_urls(packages, curr_date,
                                             py_version=py_version,
                                             platform_version=platform_version)
        if curr_wheels is not None:
            rc.append(curr_wheels)
        curr_date += timedelta(days=1)
    if reverse:
        rc.reverse()
    return rc

def get_n_prior_nightly_wheels(packages:list, n:int,
                               py_version=PYTORCH_PYTHON_VERSION, platform_version='linux_x86_64', reverse=False):
    end_date = date.today()
    start_date = end_date - timedelta(days=n)
    return get_nightly_wheels_in_range(packages, start_date, end_date,
                                       py_version=py_version, platform_version=platform_version, reverse=reverse)

def get_most_recent_successful_wheels(packages: list, pyver: str, platform: str) -> List[str]:
    """Get the most recent successful nightly wheels. Return List[str] """
    curr_date = date.today()
    date_limit = curr_date - timedelta(days=365)
    while curr_date >= date_limit:
        wheels = get_nightly_wheel_urls(packages, curr_date, py_version=pyver, platform_version=platform)
        if wheels:
            return wheels
        curr_date = curr_date - timedelta(days=1)
    # Can't find any valid pytorch package
    return None

def install_wheels(wheels):
    """Install the wheels specified in the wheels."""
    wheel_urls = list(map(lambda x: wheels[x]["wheel"], wheels.keys()))
    work_dir = Path(__file__).parent.joinpath(".data")
    work_dir.mkdir(parents=True, exist_ok=True)
    requirements_file = work_dir.joinpath("requirements.txt").resolve()
    with open(requirements_file, "w") as rf:
        rf.write("\n".join(wheel_urls))
    command = ["pip", "install", "-r", str(requirements_file)]
    print(f"Installing pytorch nightly packages command: {command}")
    subprocess.check_call(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyver", type=str, default=PYTORCH_PYTHON_VERSION, help="PyTorch Python version")
    parser.add_argument("--platform", type=str, default="linux_x86_64", help="PyTorch platform")
    parser.add_argument("--priordays", type=int, default=1, help="Number of days")
    parser.add_argument("--reverse", action="store_true", help="Return reversed result")
    parser.add_argument("--packages", required=True, type=str, nargs="+", help="List of package names")
    parser.add_argument("--install-nightlies", action="store_true",
                        help="Install the most recent successfully built nightly packages")
    args = parser.parse_args()

    if args.install_nightlies:
        wheels = get_most_recent_successful_wheels(args.packages, args.pyver, args.platform)
        assert wheels, f"We do not find any successful pytorch nightly build of packages: {args.packages}."
        print(f"Found pytorch nightly wheels: {wheels} ")
        install_wheels(wheels)
        exit(0)

    wheels = get_n_prior_nightly_wheels(packages=args.packages,
                                        n=args.priordays,
                                        py_version=args.pyver,
                                        platform_version=args.platform,
                                        reverse=args.reverse)
    for wheelset in wheels:
        for pkg in wheelset:
            print(f"{pkg}-{wheelset[pkg]['version']}: {wheelset[pkg]['wheel']}")
