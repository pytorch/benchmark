import os
import re
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

torch_wheel_cuda_version = "cu113"
torch_wheel_python_version = "cp38"
torch_wheel_platform = "linux_x86_64"
torch_wheel_nightly_base = f"https://download.pytorch.org/whl/nightly/{torch_wheel_cuda_version}/"
torch_nightly_wheel_index = f"https://download.pytorch.org/whl/nightly/{torch_wheel_cuda_version}/torch_nightly.html"


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
def get_wheel_index_data(py_version, platform_version, url=torch_nightly_wheel_index):
    """
    """
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    links = soup.find_all('a')
    data = defaultdict(dict)
    for link in soup.find_all('a'):
        pkg, version, py, py_m, platform = re.search("([a-z_]*)-(.*)-(.*)-(.*)-(.*)\.whl", link.text).groups()
        if py == py_version and platform == platform_version:
            full_url = os.path.join(torch_wheel_nightly_base, link.text)
            data[pkg][version] = full_url
    return data

def get_nightly_wheel_urls(packages:list, date:date,
                           py_version=torch_wheel_python_version, platform_version=torch_wheel_platform):
    """Gets urls to wheels for specified packages matching the date, py_version, platform_version
    """
    date_str = f"{date.year}{date.month:02}{date.day:02}"
    data = get_wheel_index_data(py_version, platform_version)

    dbg_key = None 
    versions = []
    for pkg in packages:
        pkg_versions = data[pkg]
        keys = [key for key in pkg_versions if date_str in key]
        assert len(keys) <= 1, "Did not expect multiple versions matching a date"
        if len(keys) == 0:
            return None
        if pkg == 'torch':
            dbg_key = keys[0]

        full_url = pkg_versions[keys[0]]
        versions.append(full_url)
    #print(f"   \"{dbg_key}\"  \\")
    return tuple(versions)

def get_nightly_wheels_in_range(packages:list, start_date:date, end_date:date,
                                py_version=torch_wheel_python_version, platform_version=torch_wheel_platform):
    rc = []
    curr_date = start_date
    while curr_date < end_date:
        curr_wheels = get_nightly_wheel_urls(packages, curr_date,
                                             py_version=py_version,
                                             platform_version=platform_version)
        if curr_wheels is not None:
            rc.append(curr_wheels)
        curr_date += timedelta(days=1)

    return rc

def get_n_prior_nightly_wheels(packages:list, n:int,
                               py_version=torch_wheel_python_version, platform_version=torch_wheel_platform):
    end_date = date.today()
    start_date = end_date - timedelta(days=n)
    return get_nightly_wheels_in_range(packages, start_date, end_date,
                                       py_version=py_version, platform_version=platform_version)


def create_requirements_files(root: Path, packages: list, start_date: date, end_date: date,
                              py_version=torch_wheel_python_version, platform_version=torch_wheel_platform):
    root = Path(root)
    curr_date = start_date
    while curr_date < end_date:
        curr_wheels = get_nightly_wheel_urls(packages, curr_date,
                                             py_version=py_version,
                                             platform_version=platform_version)
        if curr_wheels is not None:
            filename = root / f"requirements-{str(curr_date)}.txt"
            with open(filename, 'w') as f:
                for pkg in curr_wheels:
                    f.write(pkg + '\n')
        curr_date += timedelta(days=1)

def parse_date_str(s: str):
    return datetime.strptime(s, '%Y%m%d').date() 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['create_requirements'])
    parser.add_argument('--start_date', type=parse_date_str)
    parser.add_argument('--end_date', default=date.today(),
                        type=parse_date_str)
    parser.add_argument('--packages', nargs='+', default=['torch', 'torchvision', 'torchtext'])
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    if args.action == 'create_requirements':
        assert args.start_date is not None
        assert args.end_date is not None
        assert args.output_dir is not None
        assert not os.path.exists(args.output_dir), "provide non-existing output dir"
        os.mkdir(args.output_dir)
        create_requirements_files(args.output_dir, args.packages, args.start_date, args.end_date)
