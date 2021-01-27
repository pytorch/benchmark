import os
import re
import requests
from datetime import date, timedelta
from bs4 import BeautifulSoup
from collections import defaultdict

torch_wheel_nightly_base ="https://download.pytorch.org/whl/nightly/cu102/" 
torch_nightly_wheel_index = "https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html" 
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
    if override_file:
        with open(override_file) as f:
            data = f.read()
    else:
        r = requests.get(url)
        r.raise_for_status()
        data = r.text
    soup = BeautifulSoup(data, 'html.parser')
    data = defaultdict(dict)
    for link in soup.find_all('a'):
        pkg, version, py, py_m, platform = re.search("([a-z]*)-(.*)-(.*)-(.*)-(.*)\.whl", link.text).groups()
        if py == py_version and platform == platform_version:
            full_url = os.path.join(torch_wheel_nightly_base, link.text)
            data[pkg][version] = full_url
    return data

def get_nightly_wheel_urls(packages:list, date:date,
                           py_version='cp37', platform_version='linux_x86_64'):
    """Gets urls to wheels for specified packages matching the date, py_version, platform_version
    """
    date_str = f"{date.year}{date.month:02}{date.day:02}"
    data = get_wheel_index_data(py_version, platform_version)

    rc = {}
    for pkg in packages:
        pkg_versions = data[pkg]
        keys = [key for key in pkg_versions if date_str in key]
        assert len(keys) <= 1, "Did not expect multiple versions matching a date"
        if len(keys) == 0:
            return None
        full_url = pkg_versions[keys[0]]
        rc[pkg] = {
                        "version": keys[0],
                        "wheel": full_url,
        }
    return rc

def get_nightly_wheels_in_range(packages:list, start_date:date, end_date:date,
                                py_version='cp37', platform_version='linux_x86_64'):
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
                               py_version='cp37', platform_version='linux_x86_64'):
    end_date = date.today()
    start_date = end_date - timedelta(days=n)
    return get_nightly_wheels_in_range(packages, start_date, end_date,
                                       py_version=py_version, platform_version=platform_version)


if __name__ == "__main__":
    from tabulate import tabulate
    # print(tabulate(get_n_prior_nightly_wheels(['torch', 'torchvision', 'torchtext'], 200)))
    wheels = get_n_prior_nightly_wheels(['torch', 'torchvision', 'torchtext'], 1)
    for wheelset in wheels:
        for pkg in wheelset:
            print(f"{wheelset[pkg]['version']}: {wheelset[pkg]['wheel']}")
            # print(f"   \"{a} {b} {c}\"  \\")