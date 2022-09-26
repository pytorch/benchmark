import argparse
import os
import json
import yaml
from pathlib import Path

WORKFLOW_LINK_TEMPLATE = "https://github.com/pytorch/benchmark/actions/runs/"

def check_env(bisection_root: str):
    "Check `bisection_root` contains bisection config file, github issue file, and result json."
    # gh-issue.md exists
    # result.json exists
    bisection_path = Path(bisection_root)
    assert os.environ["GITHUB_ENV"], f"GITHUB_ENV environment variable doesn't exist."
    assert bisection_path.is_dir(), f"Specified bisection root {bisection_path} is not a directory."
    assert bisection_path.joinpath("gh-issue.md").exists(), \
        f"Bisection directory {bisection_path} doesn't contain file gh-issue.md."
    assert bisection_path.joinpath("result.json").exists(), \
        f"Bisection directory {bisection_path} doesn't contain file result.json."
    assert bisection_path.joinpath("config.yaml").exists(), \
        f"Bisection directory {bisection_path} doesn't contain file config.yaml."

def setup_gh_issue(bisection_root: str, gh_workflow_id: str):
    bisection_path = Path(bisection_root)
    json_path = bisection_path.joinpath("result.json")
    with open(json_path, "r") as jp:
        result = jp.read()
    result = f"\nResult json: \n```\n{result}\n```"
    workflow_str = f"\nBisection workflow link: {WORKFLOW_LINK_TEMPLATE}{gh_workflow_id}\n"
    gh_issue_path = bisection_path.joinpath("gh-issue.md")
    with open(gh_issue_path, "a") as ghi:
        ghi.write(result)
        ghi.write(workflow_str)

def set_env_if_nonempty(bisection_root: str):
    bisection_path = Path(bisection_root)
    json_path = bisection_path.joinpath("result.json")
    with open(json_path, "r") as jp:
        result = json.load(jp)
    # if result is empty, no need to setup the env
    if result["result"] == []:
        return
    yaml_path = bisection_path.joinpath("config.yaml")
    with open(yaml_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    affected_pytorch_version = config["end_version"]
    fname = os.environ["GITHUB_ENV"]
    content = f"TORCHBENCH_PERF_BISECTION_NONEMPTY_SIGNAL='{affected_pytorch_version}'\n"
    with open(fname, 'a') as fo:
        fo.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bisection-root", required=True, help="Root directory of the bisection directory")
    parser.add_argument("--gh-workflow-id", required=True, help="GitHub workflow id")
    args = parser.parse_args()
    check_env(args.bisection_root)
    setup_gh_issue(args.bisection_root, args.gh_workflow_id)
    set_env_if_nonempty(args.bisection_root)
