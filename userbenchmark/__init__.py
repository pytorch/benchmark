import yaml
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

def list_userbenchmarks():
    ub_dirs = [x for x in CURRENT_DIR.iterdir() if x.is_dir() and x.joinpath('__init__.py').exists() ]
    ub_names = list(map(lambda x: x.name, ub_dirs))
    return ub_names

def get_ci_from_ub(ub_name):
    ci_file = CURRENT_DIR.joinpath(ub_name).joinpath("ci.yaml")
    if not ci_file.exists():
        return None
    with open(ci_file, "r") as ciobj:
        cicfg = yaml.safe_load(ciobj)
    ret = {}
    ret["name"] = ub_name
    ret["ci_cfg"] = cicfg
    return ret

def get_userbenchmarks_by_platform(platform):
    ub_names = list_userbenchmarks()
    cfgs = list(map(lambda x: x["name"], filter(lambda x: x and x["ci_cfg"]["platform"] == platform, map(get_ci_from_ub, ub_names))))
    return cfgs
